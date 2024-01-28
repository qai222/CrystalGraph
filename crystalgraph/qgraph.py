import abc
import itertools
import logging
from collections import defaultdict
from typing import Generator

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import to_agraph
from pymatgen.core.periodic_table import Element
from pymatgen.core.periodic_table import _pt_data
from pymatgen.core.structure import Structure, PeriodicSite, Lattice, Composition
from pymatgen.vis.structure_vtk import EL_COLORS

from crystalgraph.params import _default_CNN
from crystalgraph.utils import all_have_attributes, gm_node, multigraph_cycles, edge_hash, is_3d_parallel, import_string

_allowed_voltages = tuple(itertools.product(range(-1, 2), repeat=3))


class QGerror(Exception): pass


class QuotientGraph(metaclass=abc.ABCMeta):
    """
    A crystal quotient graph is a finite graph.
    Its nodes are chemical entities (atoms/building units) and edges are interatomic chemical bonds.
    The nx graph object used to init this must have node label "symbol" defined for every node.
    """

    def __init__(self, graph, graph_class=None, properties: dict = None, ):
        # TODO since both UQG and LQG use multigraph, graph_class may be removed here
        self.nxg = graph
        self.nxg_class = graph_class
        self.properties = properties
        self.check()

    def as_dict(self):
        return {
            "nl_data": nx.node_link_data(self.nxg),
            "graph_class": self.nxg_class.__module__ + "." + self.nxg_class.__name__,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, d: dict):
        assert import_string(d['graph_class']) == nx.MultiGraph  # TODO may change
        for link in d['nl_data']['links']:
            link['voltage'] = tuple(link['voltage'])
        return cls(
            graph=nx.node_link_graph(d['nl_data']),
            properties=d['properties']
        )

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
    The unlabelled quotient graph (UQG) is an undirected multigraph.
    """

    def __init__(self, graph: nx.MultiGraph, properties: dict = None, ):
        super().__init__(graph, graph_class=nx.MultiGraph, properties=properties)

    def check(self):
        try:
            assert isinstance(self.nxg, self.nxg_class)
            assert all_have_attributes(self.nxg, ("symbol",), element="node")
        except AssertionError:
            raise QGerror("UQG check failed!")

    def draw(self, num="uqg", pos=None, figsize=(4, 4)):
        g = self.nxg
        fig = plt.figure(num, figsize=figsize)
        edge_colors = {1: "k", 2: "r", 3: "g"}

        if pos is None:
            pos = nx.spring_layout(g, seed=42)

        try:
            edge_colors = [edge_colors[e[2]["nshare"]] for e in g.edges(data=True)]
        except KeyError:
            edge_colors = ["k" for _ in g.edges(data=True)]
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
        fig.tight_layout()
        return fig, pos

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
        return set(d["voltage"] for u, v, k, d in self.nxg.edges(data=True, keys=True)).issubset(_allowed_voltages)

    def to_uqg(self) -> UQG:
        g = nx.MultiGraph()
        for n, d in self.nxg.nodes(data=True):
            g.add_node(n, **d)
        for u, v, k in self.nxg.edges:  # strip all edge attributes
            g.add_edge(u, v, key=k)
        return UQG(g, self.properties)

    def draw_graphviz(self, filename="multi.png", ipython=False):
        # use graphviz to plot graph with parallel edges
        g = nx.MultiDiGraph()

        for n, d in self.nxg.nodes(data=True):
            g.add_node(n, label="{}{}".format(n, d["symbol"]))

        edge_colors = {1: "gray", 2: "r", 3: "g"}
        for u, v, k, d in self.nxg.edges(data=True, keys=True):
            voltage = d["voltage"]
            if voltage == (0, 0, 0):
                label = ""
            else:
                label = "".join(str(i) for i in voltage)
            n1, n2 = d["direction"]
            if "nshare" in d:
                e_color = edge_colors[d["nshare"]]
                g.add_edge(n1, n2, key=k, label=label, color=e_color)
            else:
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

    def voltage_sum_cycle(self, edge_list, cycle) -> tuple:
        multigraph = self.nxg
        voltage = np.zeros(3, dtype=int)
        assert len(edge_list) == len(cycle)
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

    def voltage_sum_path(self, edge_list, path=None):
        multigraph = self.nxg
        voltage = np.zeros(3, dtype=int)
        if path is None:
            path = [e[0] for e in edge_list]
            path.append(edge_list[-1][1])
        assert len(edge_list) + 1 == len(path)
        for i, (u, v, k) in enumerate(edge_list):
            head = path[i]
            tail = path[i + 1]
            direction = multigraph.edges[(u, v, k)]["direction"]
            edge_voltage = np.array(multigraph.edges[(u, v, k)]["voltage"], dtype=int)
            if direction[0] == head and direction[1] == tail:
                voltage += edge_voltage
            elif direction[0] == tail and direction[1] == head:
                voltage -= edge_voltage
            else:
                raise RuntimeError("edge direction in edge attribute does not align with the edge, this is impossible")
        return tuple(voltage)

    def is_equivalent(self, other):
        """
        this is the 'narrower' definition of equivalence based on cycle voltage
        #TODO optimize performance

        Note:
        While it is claimed that for 3D crystals two LQGs of the same net (crystallographic net) cannot have
        non-isomorphic UQGs, this may not be true for 1D: Considering two polymers
        1. A - B - A - B ...
           |   |   |   |
           A - B - A - B ...
        2. A   B - A   B ...
           | X |   | X |
           A   B - A   B ...
        One can go from 1. to 2. by "twisting" every other unit vertically,
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
                voltage_sum1 = self.voltage_sum_cycle(edge_list1, cycle1)
                voltage_sum2 = other.voltage_sum_cycle(edge_list2, cycle2)
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
            g.add_node(i, symbol=n.species_string, frac_coords=s[i].frac_coords)
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
                    # v_cart = s.lattice.get_cartesian_coords(v_frac)
                    g.add_edge(n, neighbor["site_index"],
                               # v_frac=v_frac, v_cart=v_cart,
                               direction=(n, neighbor["site_index"]), voltage=voltage)
                    visited_voltage_edges.append(voltage_edge)
        if prop is None:
            prop = dict()
        prop["lattice"] = s.lattice
        return cls(g, properties=prop)

    def to_structure(self, lattice=None, barycentric=False, barycentric_dim=3) -> Structure:

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

        sites = []
        if barycentric:
            self.barycentric_placement(barycentric_dim)
        else:
            assert all_have_attributes(self.nxg, ("frac_coords",), "node")
            # TODO use edge vectors to embed

        for n, d in self.nxg.nodes(data=True):
            site_symbol = d["symbol"]
            if site_symbol not in _pt_data.keys():
                es = [e.name for e in Composition(site_symbol).elements if e.name not in ["O", "H"]]
                es = sorted(es, key=lambda x: Element(x).Z, reverse=True)
                site_symbol = es[0]
                logging.warning(
                    "this node has a symbol not in the periodic table, we think it is a PBU and use its heaviest element: {}".format(
                        self.symbols[n]))
            if barycentric:
                s = PeriodicSite(site_symbol, d["barypos"], lattice=lattice)
            else:
                s = PeriodicSite(site_symbol, d["frac_coords"], lattice=lattice)
            sites.append(s)
        return Structure.from_sites(sites)

    def bu_contraction(
            self,
            allowed_terminals=("O", "H", "F", "Cl", "Br", "I"),
            allowed_centers=("Si", "B", "C", "O", "H", "N", "F", "P", "S", "Cl", "As", "Se", "Br", "I",),
            allow_metal_center=True,

    ):
        """
        contract atom nodes to building units
        a building unit is currently defined as a k>=3 star of limited symbols, so it's actually polyhedron contraction
        this is coded specifically for oxide/oxysalt, be cautious when dealing with other chemical systems
        #TODO maybe better to use a general graph definition so carboxylic are included, see `The Journal of Chemical Physics 154.18 (2021): 184708.`
        """
        if allow_metal_center:
            star_element_check = lambda center, outers: (center in allowed_centers or Element(center).is_metal) and set(
                outers).issubset(set(allowed_terminals)) and len(outers) >= 3
        else:
            star_element_check = lambda center, outers: center in allowed_centers and set(outers).issubset(
                set(allowed_terminals)) and len(outers) >= 3

        stars = []
        for n in self.nxg.nodes:
            nbs = list(nx.neighbors(self.nxg, n))
            center_element = self.symbols[n]
            nb_elements = [self.symbols[nb] for nb in nbs]
            if star_element_check(center=center_element, outers=nb_elements):
                star = [n] + nbs
                star = tuple(star)
                stars.append(star)  # the first node is the center
        n_stars = len(stars)

        # helpers
        star_center_to_star = {s[0]: s for s in stars}
        star_index_to_center = {i: s[0] for i, s in enumerate(stars)}
        star_intersection_table = defaultdict(dict)
        for i in range(n_stars):
            for j in range(n_stars):
                if i == j: continue
                star_intersection_table[i][j] = set(stars[i]).intersection(set(stars[j]))

        # a set of checks
        # 1. there must be at least one star
        assert n_stars, "there is no allowed star"
        # 2. two centers cannot be directly connected
        for i, j in itertools.combinations(range(n_stars), r=2):
            if i == j:
                continue
            assert not self.nxg.has_edge(
                star_index_to_center[i], star_index_to_center[j],
            ), "two centers cannot be connected by one edge, exception found: {} and {}".format(stars[i], stars[j])
        # 3. a star must have at least one neighboring star (sharing at least one terminal node),
        # this excludes e.g. peroxides
        nshare_list = []
        for i in range(n_stars):
            nshare = 0
            for j in range(n_stars):
                if i == j:
                    continue
                if len(star_intersection_table[i][j]):
                    nshare += 1
            assert nshare > 0, "the following star does not have a neighboring star: {}".format(stars[i])
            nshare_list.append(nshare)

        # 4. the union of stars should cover all sites
        # TODO is this necessarily an error? We may loose isolated sites but that's kinda fine?
        nodes_in_stars = []
        for star in stars:
            nodes_in_stars += list(star)
        assert set(nodes_in_stars) == {*self.nxg.nodes}, "some nodes do not present in at least one star"

        # helper function to convert a star defined by LQG nodes to a string node in BU-LQG
        star_to_string_node = lambda nodelist: "-".join([str(n) for n in nodelist])

        # the resulting BU-LQG
        res = nx.MultiGraph()

        # add nodes to BU-LQG
        for star in stars:
            star_symbol = Composition(" ".join([self.symbols[n] for n in star])).formula
            star_node = star_to_string_node(star)
            try:
                center_node_frac_coords = self.nxg.nodes[star[0]]["frac_coords"]
                res.add_node(star_node, symbol=star_symbol, center=star[0], center_symbol=self.symbols[star[0]],
                             frac_coords=center_node_frac_coords)
            except KeyError:
                res.add_node(star_node, symbol=star_symbol, center=star[0], center_symbol=self.symbols[star[0]])

        # find edges for BU-LQG
        # keep in mind LQG is in principle a multigraph, always consider parallel edges
        # 1. find all length-2 *edge* paths from i-center to j-center
        # 2. calculate its voltage sum v_ij
        # 3. if two paths have the same voltage, group them together
        # 4. for each group, add an edge and assign the voltage with direction i->j
        star_ijs = [(i, j) for i, j in itertools.combinations(range(n_stars), r=2) if star_intersection_table[i][j]]
        groups = {}
        for i, j in star_ijs:
            star_i_center = star_index_to_center[i]
            star_j_center = star_index_to_center[j]
            star_i_string_node = star_to_string_node(stars[i])
            star_j_string_node = star_to_string_node(stars[j])
            paths = list(nx.all_simple_edge_paths(self.nxg, star_i_center, star_j_center, cutoff=2))
            for edge_list in paths:
                if len(edge_list) != 2:
                    continue
                voltage = self.voltage_sum_path(edge_list)
                direction = (star_i_string_node, star_j_string_node)
                key = (voltage, direction)
                if key not in groups:
                    groups[key] = [edge_list]
                else:
                    groups[key].append(edge_list)
                # node_path = [e[0] for e in edge_list] + [edge_list[-1][1]]
        # TODO this means connections like edge_sharing or face_sharing will always be represented as one edge
        #  they can be distinguished only by the size of `edge_lists`
        for k in groups:
            v, d = k
            edge_lists = groups[k]
            res.add_edge(d[0], d[1], key=None, direction=d, voltage=v,
                         edge_lists=edge_lists, nshare=len(edge_lists))
        return LQG(res)

    def barycentric_placement(self, dim=3) -> np.ndarray:
        """
        copied from gavrog project by Olaf Delgado-Friedrichs
        https://github.com/odf/gavrog/blob/master/src/org/gavrog/joss/pgraphs/basic/PeriodicGraph.java
        #TODO it seems this does not work for 1D structures, see ROBRIK case
        """
        n = len(self)
        g = self.nxg
        adjdict = g.adj
        A = np.zeros((n, n))
        t = np.zeros((n, dim))
        vert2vid = dict(zip(g.nodes, range(len(self))))
        A[0, 0] = 1
        for i in range(1, n):
            v = list(g.nodes)[i]
            for w in adjdict[v]:
                if v == w:
                    continue
                j = vert2vid[w]
                for k in adjdict[v][w]:
                    edgedata = adjdict[v][w][k]
                    direction = edgedata["direction"]
                    voltage = edgedata["voltage"]
                    if direction == (v, w):
                        this_voltage = np.array(voltage)
                    else:
                        this_voltage = -np.array(voltage)
                    A[i][j] -= 1
                    A[i][i] += 1
                    t[i] += this_voltage
        p = np.linalg.solve(A, t)
        attrs = {n: {"barypos": np.round(pos, 3)} for pos, n in zip(p, g.nodes)}
        nx.set_node_attributes(self.nxg, attrs)
        return p


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
