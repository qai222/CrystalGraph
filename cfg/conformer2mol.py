import copy
import itertools
import re
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
from pymatgen.core.periodic_table import Element
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

"""
copied from ocelot.routines.conformerparser

based on Jan H. Jensen's implementation in [xyz2mol](https://github.com/jensengroup/xyz2mol)

# TODO this is based on atom connectivity, would be good to include bond information so bond type can be determined such that a radical fragment can have bond type of its parent, this is why i cannot use rdkit AddHs in conformer_addh
"""


def chiral_stereo_check(mol):
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)
    return mol


def clean_charges(mol):
    Chem.SanitizeMol(mol)
    rxn_smarts = ['[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][CX3-,NX3-:5][#6,#7:6]1=[#6,#7:7]>>\
                   [#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][-0,-0:5]=[#6,#7:6]1[#6-,#7-:7]',
                  '[#6,#7:1]1=[#6,#7:2][#6,#7:3](=[#6,#7:4])[#6,#7:5]=[#6,#7:6][CX3-,NX3-:7]1>>\
                   [#6,#7:1]1=[#6,#7:2][#6,#7:3]([#6-,#7-:4])=[#6,#7:5][#6,#7:6]=[-0,-0:7]1']

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
                Chem.SanitizeMol(fragment)
                # print(Chem.MolToSmiles(fragment))
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol, fragment)

    return mol


def valence_electron(element):
    """
    count bucharge electrons based on electronic configuration
    if a subshell has > 10 electrons, this subshell is ignored
    """
    configuration = element.data["Electronic structure"]
    list_split = configuration.split('.')

    valence_electrons = 0

    for i in range(len(list_split)):
        if 'sup' in list_split[i]:
            electrons = re.search('<sup>(.*)</sup>', list_split[i])
            if not int(electrons.group(1)) >= 10:
                valence_electrons += int(electrons.group(1))

    return valence_electrons


def pmgmol_to_rdmol(pmg_mol):
    m = pmg_mol
    charge = m.charge
    numsites = len(m)
    conf = Chem.Conformer(numsites)
    coordmat = m.cart_coords
    for i in range(numsites):
        conf.SetAtomPosition(i, (coordmat[i][0], coordmat[i][1], coordmat[i][2]))

    mat = np.zeros((numsites, numsites))
    distmat = squareform(pdist(coordmat))
    for i in range(numsites):
        istring = pmg_mol[i].species_string
        irad = Element(istring).atomic_radius
        if irad is None:
            continue
        for j in range(i + 1, numsites):
            jstring = pmg_mol[j].species_string
            jrad = Element(jstring).atomic_radius
            if jrad is None:
                continue
            cutoff = (irad + jrad) * 1.30
            if 1e-5 < distmat[i][j] < cutoff:
                mat[i][j] = 1
                mat[j][i] = 1
    ap = ACParser(mat, charge, m.atomic_numbers, sani=True)
    try:
        rdmol, smiles = ap.parse(charged_fragments=False, force_single=False, expliciths=True)
    except Chem.rdchem.AtomValenceException:
        warnings.warn('AP parser cannot use radical scheme, trying to use charged frag')
        rdmol, smiles = ap.parse(charged_fragments=True, force_single=False, expliciths=True)
    rdmol.AddConformer(conf)
    return rdmol, smiles


class ACParser:

    def __init__(self, ac: np.ndarray, charge, atomnumberlist, sani=True, apriori_radicals=None):
        """
        :var self.valences_list: a list of possible bucharge assignment, valences_list[i] is one possbile way to assign jth atom
        bucharge based on  valences_list[i][j].
        :var self.atomic_valence_electrons: atomic_valence_electrons[i] is the #_of_ve of ith atom
        :var self.apriori_radicals: a dict to mark the atoms that can will have a lower bucharge in generating BO
        """
        self.apriori_radicals = apriori_radicals
        self.AC = ac.astype(int)
        self.sani = sani
        self.atomic_numbers = atomnumberlist
        self.natoms = len(self.atomic_numbers)
        self.charge = charge
        self.valences_list, self.atomic_valence_electrons, self.AC_valence = self.get_valence_info()

    def init_rdmol(self):
        mol = Chem.MolFromSmarts("[#" + str(self.atomic_numbers[0]) + "]")
        rwmol = Chem.RWMol(mol)
        for s in self.atomic_numbers[1:]:
            rwmol.AddAtom(Chem.Atom(s))
        mol = rwmol.GetMol()
        return mol

    def get_valence_info(self):
        atomic_valence = defaultdict(list)
        atomic_valence_electrons = {}
        for z in list(set(self.atomic_numbers)):
            e = Element.from_Z(z)
            atomic_valence[z] = list(set([abs(v) for v in e.common_oxidation_states]))
            atomic_valence_electrons[z] = valence_electron(e)

        AC_valence = list(
            self.AC.sum(axis=1))  # bucharge based on # of neighbors, can be considered as an element of valences_list

        # make a list of valences, e.g. for CO: [[4],[2,1]]
        valences_list_of_lists = []
        iatom = 0
        for atomicNum, valence in zip(self.atomic_numbers, AC_valence):
            # bucharge can't be smaller number of neighbourgs
            if self.apriori_radicals:
                try:
                    nradicals = self.apriori_radicals[iatom]
                except KeyError:
                    nradicals = 0
                possible_valence = [x - nradicals for x in atomic_valence[atomicNum] if x >= valence]
            else:
                possible_valence = [x for x in atomic_valence[atomicNum] if x >= valence]
            valences_list_of_lists.append(possible_valence)
            iatom += 1
        # from pprint import pprint
        # pprint(valences_list_of_lists)

        # convert [[4],[2,1]] to [[4,2],[4,1]]
        valences_list = list(itertools.product(*valences_list_of_lists))
        return valences_list, atomic_valence_electrons, AC_valence

    @staticmethod
    def getUADU(maxValence_list, valence_list):
        """
        get unsaturated atoms (UA) and degree of unsaturation (DU) between two possible assignments
        :param maxValence_list:
        :param valence_list:
        :return:
        """
        UA = []
        DU = []
        for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
            if maxValence - valence > 0:
                UA.append(i)
                DU.append(maxValence - valence)
        return UA, DU

    @staticmethod
    def get_bonds(UA, AC):
        """
        get a list of unique bond tuples (i, j) between UAs
        :param UA:
        :param AC:
        :return:
        """
        bonds = []
        for k, i in enumerate(UA):
            for j in UA[k + 1:]:
                if AC[i, j] == 1:
                    bonds.append(tuple(sorted([i, j])))
        return bonds

    @staticmethod
    def get_UA_pairs(UA, AC):
        """
        find the largest list of bonds in which all atom appears at most once
        :param UA:
        :param AC:
        :return:
        """
        bonds = ACParser.get_bonds(UA, AC)
        if len(bonds) == 0:
            return [()]

        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    @staticmethod
    def get_BO(AC, DU_init, valences, UA_pairs):
        """
        for a bucharge assignment, get BO
        BO[i][j] is the bond order between ith and jth
        AC is a BO with all single bond
        the algo is to increase bond order s.t. degree of unsaturation (DU) does not change
        notice DU is calculated based on the given valences
        :param DU_init:
        :param AC:
        :param valences:
        :param UA_pairs:
        :return:
        """
        BO = AC.copy()
        DU_save = []
        DU = copy.deepcopy(DU_init)

        while DU_save != DU:
            for i, j in UA_pairs:
                BO[i, j] += 1
                BO[j, i] += 1

            BO_valence = list(BO.sum(axis=1))
            DU_save = copy.copy(DU)
            UA, DU = ACParser.getUADU(valences, BO_valence)
            UA_pairs = ACParser.get_UA_pairs(UA, AC)[0]

        return BO, UA_pairs

    @staticmethod
    def get_atomic_charge(atomic_number, atomic_valence_electrons, BO_valence):
        """
        atomic charge from #_valence_electrons - bond_order
        #TODO test robustness
        :param atomic_number:
        :param atomic_valence_electrons:
        :param BO_valence:
        :return:
        """
        if atomic_number == 1:
            charge = 1 - BO_valence
        elif atomic_number == 5:
            charge = 3 - BO_valence
        elif atomic_number == 15 and BO_valence == 5:
            charge = 0
        elif atomic_number == 16 and BO_valence == 6:
            charge = 0
        else:
            charge = atomic_valence_electrons - 8 + BO_valence
        return charge

    @staticmethod
    def valences_not_too_large(BO, vs):
        number_of_bonds_list = BO.sum(axis=1)
        for valence, number_of_bonds in zip(vs, number_of_bonds_list):
            if number_of_bonds > valence:
                return False
            return True

    def BO_is_OK(self, BO, DU_from_AC, atomicNumList, charged_fragments, valences):
        """
        check bond order matrix based on
        :param BO:
        :param DU_from_AC: based on valences arg
        :param atomicNumList:
        :param charged_fragments:
        :param valences: bucharge assignment related to current BO
        :return:
        """

        if not self.valences_not_too_large(BO, valences):
            return False

        Q = 0  # total charge
        q_list = []
        if charged_fragments:
            BO_valences = list(BO.sum(axis=1))
            for i, atom in enumerate(atomicNumList):
                q = ACParser.get_atomic_charge(atom, self.atomic_valence_electrons[atom], BO_valences[i])
                Q += q
                if atom == 6:
                    number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                    if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                        Q += 1
                        q = 2
                    if number_of_single_bonds_to_C == 3 and Q + 1 < self.charge:
                        Q += 2
                        q = 1

                if q != 0:
                    q_list.append(q)

        if (BO - self.AC).sum() == sum(DU_from_AC) and self.charge == Q:  # and len(q_list) <= abs(charge):
            return True
        else:
            return False

    def parse_bonds(self, charged_fragments):
        """
        find the best BO
        :param charged_fragments:
        :return:
        """
        best_BO = self.AC.copy()
        for valences in self.valences_list:
            UA, DU_from_AC = self.getUADU(valences, self.AC_valence)
            if len(UA) == 0 and self.BO_is_OK(self.AC, DU_from_AC, self.atomic_numbers, charged_fragments, valences):
                return self.AC

            UA_pairs_list = self.get_UA_pairs(UA, self.AC)
            for UA_pairs in UA_pairs_list:
                BO, fin_UA_pairs = self.get_BO(self.AC, DU_from_AC, valences, UA_pairs)
                if self.BO_is_OK(BO, DU_from_AC, self.atomic_numbers, charged_fragments, valences):
                    return BO

                elif BO.sum() >= best_BO.sum() and self.valences_not_too_large(BO, valences):
                    best_BO = BO.copy()
        return best_BO

    def addBO2mol(self, rdmol, BO_matrix, charged_fragments, force_single=False):
        # based on code written by Paolo Toscani

        l = len(BO_matrix)
        l2 = len(self.atomic_numbers)
        BO_valences = list(BO_matrix.sum(axis=1))

        if l != l2:
            raise RuntimeError('sizes of adjMat ({0:d}) and atomicNumList '
                               '{1:d} differ'.format(l, l2))

        rwMol = Chem.RWMol(rdmol)

        bondTypeDict = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE
        }

        for i in range(l):
            for j in range(i + 1, l):
                bo = int(round(BO_matrix[i, j]))
                if bo == 0:
                    continue
                if force_single:
                    bt = Chem.BondType.SINGLE
                else:
                    bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
                rwMol.AddBond(i, j, bt)

        mol = rwMol.GetMol()

        if charged_fragments:
            mol = self.set_atomic_charges(mol, BO_valences, BO_matrix)
        else:
            mol = self.set_atomic_radicals(mol, BO_valences)
        return mol

    def set_atomic_charges(self, mol, BO_valences, BO_matrix):
        q = 0
        for i, atom in enumerate(self.atomic_numbers):
            a = mol.GetAtomWithIdx(i)
            charge = self.get_atomic_charge(atom, self.atomic_valence_electrons[atom], BO_valences[i])
            q += charge
            if atom == 6:
                number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    q += 1
                    charge = 0
                if number_of_single_bonds_to_C == 3 and q + 1 < self.charge:
                    q += 2
                    charge = 1

            if abs(charge) > 0:
                a.SetFormalCharge(int(charge))
        if self.sani:
            mol = clean_charges(mol)
        return mol

    def set_atomic_radicals(self, mol, BO_valences):
        # The number of radical electrons = absolute atomic charge
        for i, atom in enumerate(self.atomic_numbers):
            a = mol.GetAtomWithIdx(i)
            charge = self.get_atomic_charge(atom, self.atomic_valence_electrons[atom], BO_valences[i])
            if abs(charge) > 0:
                a.SetNumRadicalElectrons(abs(int(charge)))
        return mol

    def parse(self, charged_fragments=False, force_single=False, expliciths=True):
        BO = self.parse_bonds(charged_fragments)
        mol = self.init_rdmol()
        mol = self.addBO2mol(mol, BO, charged_fragments, force_single)
        if self.sani:
            mol = chiral_stereo_check(mol)
        smiles = Chem.MolToSmiles(mol, allHsExplicit=expliciths, isomericSmiles=True)
        m = Chem.MolFromSmiles(smiles, sanitize=self.sani)
        smiles = Chem.MolToSmiles(m, isomericSmiles=True, allHsExplicit=expliciths)

        return mol, smiles  # m is just used to get canonical smiles
