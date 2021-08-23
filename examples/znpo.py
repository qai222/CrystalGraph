from cfg.converter import sf
from cfg.fgraph import FiniteGraph, Structure

# load finite graph, this is a 1D amine-templated zinc phosphite, amine was removed
structure = Structure.from_file("ROBRIK.cif")
fg = FiniteGraph.from_structure(structure)

# we can get its smi/selfies
smi = fg.from_ac_to_smiles()
print(smi)
print(sf.encoder(smi))

# instead of using atmos, we can use primary building units as nodes
rfg = fg.reduce_oxi()
FiniteGraph.draw(rfg, "ROBRIK.png", "ROBRIK")
# how about a SELIFES using primary building units as alphabet?
