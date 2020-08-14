# Energy-Systems-Planning

Various fragments of Julia code to accompany the MSc dissertation "Energy Systems Planning: Dynamic Scenarios"

Of these:

* cutting_plane.jl is the code used to calculate the details of the example that appears in Section 2.4

* adapative_oracles_AR.jl is a small-scale implementation of the adaptive oracles Benders decomposition algorithm.  This code generates random matrices A, B and C to specify the subproblems required.  Note that with this small example, it is typical to find an exactly optimal solution in the process of finding an ε-optimal solution.  This script can be run on its own and is not needed for anything else.

* Scenario_trees.jl implements a basic scenario tree model, allowing for trees to be split at specificed nodes.  This code uses the Gadfly and Light Graphs packages to allow users to visualise the trees generated.

* Casey_Sen.jl then builds on this code to implement tools for solving problems of the form given in the example of Section 3.5 of the dissertation (and in the original paper by Casey and Sen)

The remaining files build on the existing code available at https://github.com/nimazzi/Stand_and_Adapt_Bend

* load_new_functions.jl consists of several new functions for updating and modifying data structures defined in the original codes load_functions.jl package; this should be placed in the functions subfolder with the other function definitions

* main.jl is intended to replace the main.jl used in the original code; it allows the user to specify which tree they wish to solve the master problem on and the rule used to sample off-the-tree
