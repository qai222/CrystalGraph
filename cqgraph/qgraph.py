import abc
import itertools
import logging
from typing import Generator

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import to_agraph
from pymatgen.core.structure import Structure, PeriodicSite, Lattice

from cqgraph.params import _default_CNN
from cqgraph.utils import all_have_attributes, gm_node, multigraph_cycles, edge_hash, is_3d_parallel

_allowed_voltages = tuple(itertools.product(range(-1, 2), repeat=3))


class QGerror(Exception): pass


class QuotientGraph(metaclass=abc.ABCMeta):
    """
    A crystal quotient graph is a finite graph.
    Its nodes are chemical entities (atoms/building units) and edges are interatomic chemical bonds.
    The nx graph object used to init this must have node label "symbol" defined for every node.
    """

    def __init__(self, graph, graph_class=None, properties: dict = None, ):
        self.nxg = graph
        self.nxg_class = graph_class
        self.properties = properties
        self.check()

    @abc.abstractmethod
    def check(self):
        pass

    @property
    def symbols(self) -> dict:
        """a dict s.t. symbols[node] gives symbol"""
        return nx.get_node_attributes(self.nxg, 'symbol')

    def __len__(self):
        return len(self.nxg.nodes)

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    def __repr__(self) -> str:
        header = "{}:".format(self.__class__.__name__)
        outs = []
        for n in self.nxg.nodes:
            outs.append('{}-{}'.format(n, self.symbols[n]))
        return header + "\n\t" + "; ".join(outs)


class UQG(QuotientGraph):
    """
    The unlabelled quotient graph (UQG) is an undirected, edge-unlabeled multigraph.
    """

    def __init__(self, graph: nx.MultiGraph, properties: dict = None, ):
        super().__init__(graph, graph_class=nx.MultiGraph, properties=properties)

    def check(self):
        try:
            assert isinstance(self.nxg, self.nxg_class)
            assert all_have_attributes(self.nxg, ("symbol",), element="node")
        except AssertionError:
            raise QGerror("UQG check failed!")

    def __eq__(self, other):
        """UQGs are the same by symbol-match isomorphism"""
        # return iso.is_isomorphic(self.nxg, other.nxg, node_match=iso.generic_node_match('symbol', None, eq))
        return gm_node(self.nxg, other.nxg, ("symbol",)).is_isomorphic()


class LQG(QuotientGraph):
    """
    The labelled quotient graph (LQG) is a multigraph with edges labelled by direction and voltage.
    #TODO should we have a 'VoltageGraph' base class for this?
    """

    def __init__(self, graph: nx.MultiGraph, properties: dict = None, ):
        super().__init__(graph, graph_class=nx.MultiGraph, properties=properties)

    def check(self):
        try:
            assert isinstance(self.nxg, self.nxg_class)
            assert all_have_attributes(self.nxg, ("voltage", "direction"), "edge")
            assert all_have_attributes(self.nxg, ("symbol",), "node")
            assert self.check_voltage()
        except AssertionError:
            raise QGerror("LQG check failed!")

    def check_voltage(self) -> bool:
        return set(self.nxg.edges[e]["voltage"] for e in self.nxg.edges).issubset(_allowed_voltages)

    def to_uqg(self) -> UQG:
        g = nx.MultiGraph()
        for n, d in self.nxg.nodes(data=True):
            g.add_node(n, **d)
        for u, v, k in self.nxg.edges:  # strip all edge attributes
            g.add_edge(u, v, key=k)
        return UQG(g, self.properties)

    def draw_graphviz(self, filename="multi.png", ipython=False):
        g = nx.MultiDiGraph()

        for n, d in self.nxg.nodes(data=True):
            g.add_node(n, label="{}{}".format(n, d["symbol"]))

        for u, v, k, d in self.nxg.edges(data=True, keys=True):
            voltage = d["voltage"]
            if voltage == (0, 0, 0):
                label = ""
            else:
                label = "".join(str(i) for i in voltage)
            n1, n2 = d["direction"]
            g.add_edge(n1, n2, key=k, label=label)

        g.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        g.graph['graph'] = {'scale': '3'}
        a = to_agraph(g)
        a.layout('dot')
        if ipython:
            from IPython.display import Image
            return Image(a.draw(format="png"))
        else:
            a.draw(path=filename)

    def draw(self, num, positions=None) -> (plt.Figure, np.ndarray):
        lqg = self.nxg
        if positions is None:
            try:
                pos = nx.planar_layout(lqg)
            except nx.NetworkXException:
                pos = nx.spring_layout(lqg, seed=42)
        else:
            pos = positions
        fig = plt.figure(num)
        nx.draw(
            lqg, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='white', alpha=0.9,
        )
        node_labels = {}
        for u, d in lqg.nodes.data():
            node_labels[u] = "{}{}".format(u, d["symbol"])
        edge_labels = {}
        for u, v, d in lqg.edges.data():
            direction = d["direction"]
            voltage = d["voltage"]
            if voltage == (0, 0, 0):
                edge_labels[(u, v)] = ""
            else:
                edge_labels[(u, v)] = "{}->{}: {}".format(direction[0], direction[1],
                                                          "".join(str(int(s)) for s in voltage))
        nx.draw_networkx_labels(lqg, pos, labels=node_labels, font_size=14)
        nx.draw_networkx_edge_labels(
            lqg, pos,
            edge_labels=edge_labels,
            font_color='red',
            font_size=14
        )
        return fig, positions

    def voltage_sum(self, edge_list, cycle) -> tuple:
        multigraph = self.nxg
        voltage = np.zeros(3, dtype=int)
        for i, (u, v, k) in enumerate(edge_list):
            head = cycle[i]
            if i == len(edge_list) - 1:
                tail = cycle[0]
            else:
                tail = cycle[i + 1]
            direction = multigraph.edges[(u, v, k)]["direction"]
            edge_voltage = np.array(multigraph.edges[(u, v, k)]["voltage"], dtype=int)
            if direction == (head, tail):
                voltage += edge_voltage
            else:
                voltage -= edge_voltage
        return tuple(voltage)

    def is_equivalent(self, other):
        """
        this is the 'narrower' definition of equivalence based on cycle voltage
        #TODO optimize performance

        Note:
        While it is claimed that for 3D crystals two LQGs of the same net (crystallographic net) cannot have
        non-isomorphic UQGs, this may not be true for 1D. Considering two polymers
        1. A - B - A - B ...
           |   |   |   |
           A - B - A - B ...
        2. A   B - A   B ...
           | X |   | X |
           A   B - A   B ...
        One can go from 1. to 2. by "twisting" every other unit horizontally,
        these two embeddings of the same net, however, give two non-isomorphic UQGs.
        """
        if len(self) != len(other):
            return False
        if len(self.nxg.edges) != len(other.nxg.edges):
            return False

        lqg1 = self.nxg
        lqg2 = other.nxg

        uqg1 = self.to_uqg()
        uqg2 = other.to_uqg()
        if uqg1 != uqg2:
            return False
        uqg_gm = gm_node(uqg1.nxg, uqg2.nxg, ("symbol",))
        for p in uqg_gm.isomorphisms_iter():
            edge_lists_to_cycles_1 = multigraph_cycles(lqg1)
            logging.info("checking permutation: {}".format(p))
            permutation_match = True
            for edge_list1 in edge_lists_to_cycles_1:
                cycle1 = edge_lists_to_cycles_1[edge_list1]
                cycle2 = [p[n1] for n1 in cycle1]
                # voltage sum of cycle1 in lqg1 should be the same as that in lqg2 after permutation
                edge_list2 = []
                for u, v, k in edge_list1:
                    edge_list2.append((p[u], p[v], k))
                voltage_sum1 = self.voltage_sum(edge_list1, cycle1)
                voltage_sum2 = other.voltage_sum(edge_list2, cycle2)
                logging.info(" ".join(["cycles in 1:", str(cycle1), "voltage sum:", str(voltage_sum1)]))
                logging.info(" ".join(["cycles in 2:", str(cycle2), "voltage sum:", str(voltage_sum2)]))
                if voltage_sum1 != voltage_sum2:
                    permutation_match = False
                    break
            if permutation_match:
                logging.info("cycle voltage sums identical, permutation match found")
                return True
            logging.info("this permutation does not match")
        return False

    def __eq__(self, other):
        return self.is_equivalent(other)

    @classmethod
    def from_structure(cls, s: Structure, nn_method=_default_CNN, prop=None):
        visited_voltage_edges = []
        g = nx.MultiGraph()
        for i, n in enumerate(s):
            g.add_node(i, symbol=n.species_string)
        for n, neighbors in enumerate(nn_method.get_all_nn_info(s)):
            n_image = (0, 0, 0)
            for neighbor in neighbors:
                neighbor_image = neighbor["image"]
                neighbor_index = neighbor["site_index"]
                v_frac = s[neighbor_index].frac_coords - s[n].frac_coords + np.array(neighbor_image)
                assert not np.allclose(v_frac, np.zeros(3)), "possible overlapping sites?"
                voltage_edge = VoltageEdge(v_frac, n, neighbor_index, n_image, neighbor_image)
                if voltage_edge not in visited_voltage_edges:
                    voltage = neighbor_image
                    v_cart = s.lattice.get_cartesian_coords(v_frac)
                    g.add_edge(n, neighbor["site_index"], v_frac=v_frac, v_cart=v_cart,
                               direction=(n, neighbor["site_index"]), voltage=voltage)
                    visited_voltage_edges.append(voltage_edge)
        if prop is None:
            prop = dict()
        prop["lattice"] = s.lattice
        return cls(g, properties=prop)

    def to_structure(self, lattice=None) -> Structure:

        if lattice is None:
            try:
                lattice = self.properties["lattice"]
            except KeyError:
                raise ValueError("lattice is None and cannot be found in properties!")
        else:
            try:
                assert isinstance(lattice, Lattice)
            except AssertionError:
                raise ValueError("lattice is not None but is also not a valid Lattice object!")

        n0 = list(self.nxg.nodes)
        symbols = self.symbols
        coords = dict()
        for u, v, k in nx.edge_dfs(self.nxg, source=n0):
            if u not in coords:
                coords_u = np.array([0, 0, 0])
                coords[u] = coords_u
            else:
                coords_u = coords[u]
            coords_v = coords_u + self.nxg.edges[(u, v, k)]["v_frac"]
            coords[v] = coords_v
        sites = []
        for n in symbols:
            site = PeriodicSite(symbols[n], coords[n], lattice)
            sites.append(site)
        return Structure.from_sites(sites)


class VoltageEdge:
    """an auxiliary class for generating LQGs from structures"""

    def __init__(self, vector: np.ndarray, n1: int, n2: int, n1_image: tuple, n2_image: tuple):
        self.vector = vector
        self.n1 = n1
        self.n2 = n2
        self.n1_image = n1_image
        self.n2_image = n2_image
        self.terminals = frozenset([self.n1, self.n2])

    def __repr__(self):
        return "Edge: {}, voltage: {}, vector: {}".format(sorted([self.n1, self.n2]), self.voltage, self.vector)

    def __hash__(self):
        return 42

    @property
    def length(self):
        return np.linalg.norm(self.vector)

    @property
    def voltage(self):
        terms = (self.n1, self.n2)
        imags = (self.n1_image, self.n2_image)
        a_image, b_image = (x for x, _ in sorted(zip(imags, terms), key=lambda x: x[1]))
        return tuple(a_image[i] - b_image[i] for i in range(3))

    def __eq__(self, other):
        eps = 1e-5
        is_parallel = is_3d_parallel(self.vector, other.vector, eps=eps)
        # print( self.vector, other.vector, np.cross(self.vector, other.vector), is_parallel)
        if not is_parallel:
            return False
        is_sameterminal = self.terminals == other.terminals
        if not is_sameterminal:
            return False
        is_eqlength = abs(self.length - other.length) < eps
        if not is_eqlength:
            return False
        is_eqvoltage = is_3d_parallel(self.voltage, other.voltage) or self.voltage == other.voltage == (0, 0, 0)
        if not is_eqvoltage:
            return False
        return True


class LQGeq:
    """generating equivalent LQGs from the multigraph"""

    @staticmethod
    def generator_label_and_direction(lqg: LQG) -> Generator:
        nxg = lqg.nxg
        labelled_edges = [e for e in nxg.edges if nxg.edges[e]["voltage"] != (0, 0, 0)]
        for nedge in range(0, len(labelled_edges) + 1):
            if nedge == 0:
                yield LQG(nxg.copy())
                continue
            edge_sets = itertools.combinations(labelled_edges, nedge)
            for edge_set in edge_sets:
                enumerated = nxg.copy()
                for e in edge_set:
                    enumerated.edges[e]["voltage"] = tuple(-v for v in enumerated.edges[e]["voltage"])
                    direction = enumerated.edges[e]["direction"]
                    if direction[0] != direction[1]:
                        enumerated.edges[e]["direction"] = (direction[1], direction[0])
                try:
                    yield LQG(enumerated)
                except QGerror:  # skip those failed check
                    continue

    @staticmethod
    def generator_coordination_system(
            lqg: LQG,
            allowed_origin_shifts=tuple(itertools.product(range(0, 2), repeat=3)),
            allowed_basis_vector_matrices=None,  # TODO this depends on lattice symmetry, currently not implemented

    ):
        nxg = lqg.nxg
        if allowed_basis_vector_matrices is None:
            allowed_basis_vector_matrices = [np.eye(3, dtype=int), ]
        else:
            allowed_basis_vector_matrices = [np.linalg.inv(a) for a in
                                             allowed_basis_vector_matrices]

        have_seen = []
        for m in allowed_basis_vector_matrices:
            for origin_shifts in itertools.product(allowed_origin_shifts, repeat=len(nxg.nodes)):
                enumerated = nxg.copy()
                node2shift = dict(zip(enumerated.nodes, origin_shifts))
                for e in enumerated.edges:
                    voltage = enumerated.edges[e]["voltage"]
                    direction = enumerated.edges[e]["direction"]
                    voltage = np.matmul(m, np.array(voltage))
                    head = direction[0]
                    tail = direction[1]
                    voltage = voltage + node2shift[head] - node2shift[tail]
                    enumerated.edges[e]["voltage"] = tuple(voltage)
                ehash = edge_hash(enumerated)
                if ehash not in have_seen:
                    have_seen.append(ehash)
                    try:
                        yield LQG(enumerated)
                    except QGerror:
                        continue
        # print(found, total)

    @staticmethod
    def permutation(lqg: LQG, p: dict):
        nxg = lqg.nxg
        permuted_lqg = nx.MultiGraph()
        for n in nxg.nodes:
            g2_node = p[n]
            permuted_lqg.add_node(g2_node, symbol=nxg.nodes[n]["symbol"])
        for e in nxg.edges:
            u, v, k = e
            voltage = nxg.edges[e]["voltage"]
            direction = nxg.edges[e]["direction"]
            new_direction = (p[direction[0]], p[direction[1]])
            permuted_lqg.add_edge(p[u], p[v], voltage=voltage, direction=new_direction)
        return LQG(permuted_lqg)


