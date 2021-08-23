import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import _pt_data
from pymatgen.core.structure import Structure, Composition, Element, Lattice, PeriodicSite
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.vis.structure_vtk import EL_COLORS


def get_dist_and_trans(lattice: Lattice, fc1, fc2):
    """
    get the shortest distance and corresponding translation vector between two frac coords
    """
    v, d2 = pbc_shortest_vectors(lattice, fc1, fc2, return_d2=True)
    if len(np.array(fc1).shape) == 1:
        fc = lattice.get_fractional_coords(v[0][0]) + fc1 - fc2
    else:
        fc = np.zeros((len(fc1), len(fc2), 3))
        for i in range(len(fc1)):
            for j in range(len(fc2)):
                fc_vector = np.dot(v[i][j], lattice.inv_matrix)
                fc[i][j] = fc_vector + fc1[i] - fc2[j]
    return np.sqrt(d2), fc


_default_CNN = CrystalNN(
    weighted_cn=False,
    cation_anion=False,
    distance_cutoffs=(0.5, 1),
    x_diff_weight=3.0,
    porous_adjustment=True,
)


class FiniteGraph:

    def __init__(self, graph: nx.Graph, lattice: Lattice):
        self.graph = graph
        self.lattice = lattice
        self.nnodes = len(self.graph.nodes)
        self.nedges = len(self.graph.edges)
        assert self.nnodes == len(nx.get_node_attributes(self.graph, "symbol"))
        assert self.nedges == len(nx.get_edge_attributes(self.graph, "v_frac"))
        assert self.nedges == len(nx.get_edge_attributes(self.graph, "v_cart"))

    def assgin_bondtype_dummy(self):
        from rdkit.Chem.rdchem import BondType
        g = self.graph.copy()
        attrs = dict()
        for e in g.edges:
            attrs[e] = {"bondtype": BondType.SINGLE}
        nx.set_edge_attributes(g, attrs)
        return g

    def from_ac_to_smiles(self, charge=0, return_rdmol=False):
        from cfg.conformer2mol import ACParser, Chem
        import warnings
        ac = np.zeros((len(self.graph), len(self.graph)))
        symbols = nx.get_node_attributes(self.graph, "symbol")
        for n in self.graph.nodes:  # assume nodes start from 0 and are continuous
            for nb in self.graph.neighbors(n):
                ac[n][nb] = 1
                ac[nb][n] = 1
        atmoic_numbers = [Element(symbols[n]).Z for n in self.graph.nodes]
        ap = ACParser(ac, charge, atmoic_numbers)
        try:
            rdmol, smiles = ap.parse(charged_fragments=False, force_single=False, expliciths=True)
        except Chem.rdchem.AtomValenceException:
            warnings.warn('AP parser cannot use radical scheme, trying to use charged frag')
            rdmol, smiles = ap.parse(charged_fragments=True, force_single=False, expliciths=True)
        if return_rdmol:
            return rdmol, smiles
        else:
            return smiles

    @staticmethod
    def nodes2formula(symbols, nodes):
        f = " ".join([symbols[n] for n in nodes])
        return Composition(f).alphabetical_formula

    @staticmethod
    def draw(g: nx.Graph, saveas, title=None):
        edge_colors = {1: "k", 2: "r", 3: "g"}

        pos = nx.spring_layout(g)
        try:
            edge_colors = [edge_colors[e[2]["nshare"]] for e in g.edges(data=True)]
        except KeyError:
            edge_colors = ["k" for e in g.edges(data=True)]
        node_colors = []
        for n in g.nodes(data=True):
            symbol = n[1]["symbol"]
            if symbol not in _pt_data.keys():
                es = [e.name for e in Composition(symbol).elements if e.name not in ["O", "H"]]
                es = sorted(es, key=lambda x: Element(x).Z, reverse=True)
                symbol = es[0]
            node_colors.append('#{:02x}{:02x}{:02x}'.format(*EL_COLORS['Jmol'][symbol]))
        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(g, pos, labels=nx.get_node_attributes(g, "symbol")),
        nx.draw_networkx_edges(g, pos, edgelist=g.edges, edge_color=edge_colors, arrows=False)
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])
        if title:
            plt.title(title)
        plt.savefig(saveas)
        plt.clf()

    def reduce_oxi(self, leaf_elements=("O", "H")):
        reduced_graph = nx.Graph()
        symbols = nx.get_node_attributes(self.graph, "symbol")
        for n in self.graph:
            if symbols[n] not in leaf_elements:
                mo_unit = tuple([n] + list(nx.neighbors(self.graph, n)))
                reduced_graph.add_node(mo_unit, symbol=FiniteGraph.nodes2formula(symbols, mo_unit))

        for node_i, node_j in itertools.combinations(reduced_graph.nodes, 2):
            nshare = len(set(node_i[1:]).intersection(set(node_j[1:])))
            if nshare > 0:
                reduced_graph.add_edge(node_i, node_j, nshare=nshare)
        return reduced_graph

    @classmethod
    def from_structure(cls, s: Structure, nn_method=_default_CNN):
        # assume this is a connected structure
        pmg_graph = nn_method.get_bonded_structure(s).graph
        pmg_graph = nx.Graph(pmg_graph).to_undirected()

        finite_graph = nx.Graph()
        for n in pmg_graph.nodes:
            finite_graph.add_node(n, symbol=s[n].species_string)
        for n1, n2 in pmg_graph.edges:
            edata = pmg_graph.edges[(n1, n2)]
            v_frac = s[n2].frac_coords - s[n1].frac_coords + np.array(edata["to_jimage"])
            v_cart = s.lattice.get_cartesian_coords(v_frac)
            bondlength = np.linalg.norm(v_cart)
            finite_graph.add_edge(n1, n2, v_frac=v_frac, v_cart=v_cart, bondlength=bondlength)
        return cls(finite_graph, s.lattice)

    def to_structure(self):
        n0 = list(self.graph.nodes)[0]
        symbols = nx.get_node_attributes(self.graph, "symbol")
        coords = dict()
        for u, v in nx.dfs_edges(self.graph, source=n0):
            if u not in coords:
                coords_u = np.array([0, 0, 0])
                coords[u] = coords_u
            else:
                coords_u = coords[u]
            coords_v = coords_u + self.graph.edges[(u, v)]["v_frac"]
            coords[v] = coords_v
        sites = []
        for n in symbols:
            site = PeriodicSite(symbols[n], coords[n], self.lattice)
            sites.append(site)
        return Structure.from_sites(sites)
