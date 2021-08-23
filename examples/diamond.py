from cfg.converter import graph2smiles, graph2selfies, sf, smiles2graph, selfies2graph
from cfg.fgraph import FiniteGraph, Structure, nx

structure = Structure.from_file("diamond.cif")
fg = FiniteGraph.from_structure(structure)

# print bond length
symbols = nx.get_node_attributes(fg.graph, "symbol")
bonds = []
for n1, n2, edata in fg.graph.edges(data=True):
    bonds.append((symbols[n1], symbols[n2], round(edata["bondlength"], 5)))
print(set(bonds))  # {('C', 'C', 1.54447)}

# draw the finite graph, we expect each carbon is bonded to 4 other carbon atoms
FiniteGraph.draw(fg.graph, "diamond.png", "diamond")

# we need to assgin bondtype to get smiles/selfies, we know we only have single bonds...
finite_graph = fg.assgin_bondtype_dummy()
print(list(finite_graph.edges(data=True))[0][2]["bondtype"])  #SINGLE

# now we can print out string representations
smi = graph2smiles(finite_graph)
print(smi)  # C123C45C67C18C41C26C58C371
selfies = graph2selfies(finite_graph)
print(selfies)  # [C][C][C][C][Ring1][Ring2][C][Ring1][Ring2][C][Ring1][Branch1_2][Ring1][Ring2][C][Ring1][Branch1_2][Ring1][Ring2][C][Ring1][Branch2_1][Ring1][Branch1_2][Ring1][Ring2]

# we can also try to figure out bondtype assignment based on atom connectivity
smi = fg.from_ac_to_smiles()
print(smi)
selfies = sf.encoder(smi)
print(selfies)

# going back from string to graph, you cannot get back to CIF as string doesn't tell you coordinates
re_finite_graph = smiles2graph(smi)
FiniteGraph.draw(re_finite_graph, "diamond_re.png", "diamond")

