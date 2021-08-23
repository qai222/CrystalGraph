import networkx as nx
import selfies as sf
from pymatgen.analysis.local_env import CrystalNN
from rdkit import Chem

_default_CNN = CrystalNN(
    weighted_cn=False,
    cation_anion=False,
    distance_cutoffs=(0.5, 1),
    x_diff_weight=3.0,
    porous_adjustment=True,
)


def smiles2graph(smiles: str):
    """
    return a nx.Graph obj from smiles
    copied from https://caer200.github.io/ocelot_api/_modules/ocelot/schema/graph.html
    """
    m = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(m)
    # if 'H' in smiles:  # explicit hydrogens
    #     rdmol = m
    # else:
    #     rdmol = Chem.AddHs(m)
    rdmol = m
    g = nx.Graph()
    for atom in rdmol.GetAtoms():
        g.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   )
    for bond in rdmol.GetBonds():
        g.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bondtype=bond.GetBondType(),
        )
    return g


def graph2smiles(g: nx.Graph):
    rdmol = Chem.RWMol()
    for n, ndata in g.nodes(data=True):
        aid = rdmol.AddAtom(Chem.Atom(ndata["symbol"]))
        assert aid == n

    for n1, n2, edata in g.edges(data=True):
        rdmol.AddBond(n1, n2, edata["bondtype"])
    m = rdmol.GetMol()
    return Chem.MolToSmiles(m)


def selfies2graph(selfies: str):
    smiles = sf.decoder(selfies)
    return smiles2graph(smiles)


def graph2selfies(g: nx.Graph):
    smiles = graph2smiles(g)
    return sf.encoder(smiles)

