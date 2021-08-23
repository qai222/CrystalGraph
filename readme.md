Finite graph for crystals
--
A proof of concept using finite graph to 
describe crystal structure.

The underlying periodic graph (infinitely large) 
of a crystal can be described by a finite graph
and a set of real space vectors.

Some examples can be found in the folder [examples](./examples),
specifically:
1. [diamond.py](examples/diamond.py) converts the CIF of
diamond structure to a SMILES/SELFIES string, and
produces a 2D representation of the finite graph.
2. [znpo.py](examples/znpo.py) looks at a 1D amine-templated
oxide structure, instead of atomic sites,
primary building units are used as nodes of the finite
graph. We note this finite graph represents a series of
1D zinc phosphite/phosphate structures.

### Dependencies
```
conda install -c conda-forge rdkit
conda install -c conda-forge pymatgen
pip install selfies
```

### Example: AB polymer 
Let's say we have a polymer like this:`...A=B-A=B...`

As a crystal its unit cell is `[-A=B]`

To represent this unit cell we can create a graph:
```
NODES:  \\ what's inside the unit cell?
	A, B
EDGES:  \\ for each node, what are their neighbors?
	e1, {"head": A, "tail": B, "real space direction": ">", "type": "double"}
	e2, {"head": A, "tail": B, "real space direction": "<", "type": "single"}
	e3, {"head": B, "tail": A, "real space direction": "<", "type": "double"}
	e4, {"head": B, "tail": A, "real space direction": ">", "type": "single"}
```
Note how e1 and e2 are parallel edges 
(same head and tail, thus requires a multigraph, 
for close-packing structures we would have self-loops 
). 

We can get the original crystal 
(the infinite, periodic graph) by drawing paths based on
this *finite* graph (loop over the edges n times), starting from A:
```
e0:     A
e1:     A=B
e2:   B-A=B
e3: A=B-A=B
e4: A=B-A=B-A
e1: A=B-A=B-A=B
...
```
For more details regarding the finite graph, 
see [Sunada2012](https://link.springer.com/article/10.1007/s11537-012-1144-4) 
where this is defined as the "fundamental finite graph" of a crystal. 

### Suggestions for string representations
1. Allow parallel edges and self-loops. 
I don't know how to do this in SMILES. 
For SELFIES it seems we can just add two characters with `Q=0` and `Q=-1`,
i.e. parallel edges are rings with size 1 and self-loops are rings with size 0.
2. For inorganic structures we can extend alphabet to include building units, see [znpo.py](examples/znpo.py).
3. From CIF to string, bond type has to be determined for each edge. This is *largely* solved for organics 
but I'm not sure there is a robust method for inorganic structures, especially when hydrogen atoms are missing.
4. Strictly speaking, finite graph is derived for a specific cell. That means two cells representing the same crystal could give two different finite graphs. 
So we need to either start from the most reduced cell, or find a way to reduce the finite graph (or the string representation itself).
