Crystal Graph
--
Graphs for bond topology of crystal structures.

The underlying periodic graph (infinitely large) 
of a crystal can be described by a finite graph
and a set of real space vectors.

### Install
You need `pymatgen` installed.

```
pip install crystalgraph
```

### Usage
Some examples can be found in the folder [examples](./examples),
specifically:
1. [qg_base.ipython](examples/qg_base.ipynb) describes basic usage of quotient graph class.
2. [qg_cif.ipython](examples/qg_cif.ipynb) convert CIFs to quotient graphs, and contract atom nodes to building units.

### Suggestions for string representations
1. Allow parallel edges and self-loops. 
I don't know how to do this in SMILES. 
For SELFIES it seems we can just add two characters with `Q=1` and `Q=0`,
i.e. parallel edges are rings of size 2 and self-loops are rings of size 1.
2. For inorganic structures we can extend alphabet to include building units.
3. From CIF to string, bond type has to be determined for each edge. This is *largely* solved for organics 
but I'm not sure there is a robust method for inorganic structures, especially when hydrogen atoms are missing.
4. Strictly speaking, finite graph is derived for a specific cell. That means two cells representing the same crystal could give two different finite graphs. 
So we need to either start from the most reduced cell, or find a way to reduce the finite graph (or the string representation itself).
