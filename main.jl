cd(dirname(@__FILE__))
include("./functions/load_inputfunctions.jl")
(J,δ) =(1000,0.1);
(cs,al,q,w) =  (3,3,1.0,1);
include("./functions/load_stuff.jl")
include("./functions/load_newfunctions.jl")

str = "Benders decomposition algorithm on a subtree of the full tree";
println("")
println("*"^length(str))
println(str)
println("*"^length(str))

st = get_subtree();
r = get_Iexrule();

if (r == 1)
    N = rule1_params();
    (B,S) = solve_subtree_Benders!(st,N);
elseif (r == 2)
    γ = rule2_params();
    (B,S) = solve_subtree_Benders!(st,γ);
elseif (r == 3)
    (A,α) = rule3_params();
    (B,S) = solve_subtree_Benders!(st,A,α);
elseif (r == 4)
    (B,S) = solve_subtree_Benders!(st);
end

println("")

#

print("x0 at node 0 is ")
println(round.(value.(B.rmp.m[:x0][:,1]);digits=1))
println()
print("x0 at node 14 is")
println(round.(value.(B.rmp.m[:x0][:,15]);digits=1))
println()
println("Objective value is $(JuMP.objective_value(B.rmp.m))")
println()


# temporary, just for testing purposes:

nodeA = value.(B.rmp.m[:x0][:,1]);
nodeB = value.(B.rmp.m[:x0][:,15]);

B_solved = deepcopy(B);
B1 = expand_subtree(B_solved,1)
Lt = JuMP.objective_value(B1.rmp.m);
println("Lower bound on tree 1 without fixing nodes is $(Lt)")
#println("")

#print("x0 at node 0 is ")
#println(round.(value.(B1.rmp.m[:x0][:,1]);digits=1))
#println()
#println("x0 at node 13 is")
#println(round.(value.(B1.rmp.m[:x0][:,14]);digits=1))
#println()
#println("x0 at node 14 is")
#println(round.(value.(B1.rmp.m[:x0][:,15]);digits=1))
#println()
#println("x0 at node 15 is")
#println(round.(value.(B1.rmp.m[:x0][:,16]);digits=1))
#println()
#println()

B_solved = deepcopy(B);
B1 = expand_subtree(B_solved,1,nodeA,nodeB);
Lf = JuMP.objective_value(B1.rmp.m);
println("Lower bound on tree 1 after fixing nodes is $(Lf)")

println("Impact on tree 1 of fixing nodes to deterministic value is $((Lf-Lt)/Lt)")
println("")

B_solved = deepcopy(B);
B2 = expand_subtree(B_solved,2)
Lt = JuMP.objective_value(B2.rmp.m);
println("Lower bound on tree 2 without fixing nodes is $(Lt)")

#print("x0 at node 0 is ")
#println(round.(value.(B2.rmp.m[:x0][:,1]);digits=1))
#println()
#println("x0 at node 11 is")
#println(round.(value.(B2.rmp.m[:x0][:,12]);digits=1))
#println()
#println("x0 at node 14 is")
#println(round.(value.(B2.rmp.m[:x0][:,15]);digits=1))
#println()
#println("x0 at node 17 is")
#println(round.(value.(B2.rmp.m[:x0][:,18]);digits=1))
#println()
#println()

B_solved = deepcopy(B);
B2 = expand_subtree(B_solved,2,nodeA,nodeB);
Lf = JuMP.objective_value(B2.rmp.m);
println("Lower bound on tree 2 after fixing nodes is $(Lf)")

println("Impact on tree 2 of fixing nodes to deterministic value is $((Lf-Lt)/Lt)")
println("")

B_solved = deepcopy(B);
B3 = expand_subtree(B_solved,3)
Lt = JuMP.objective_value(B3.rmp.m);
println("Lower bound on tree 3 without fixing nodes is $(Lt)")

#print("x0 at node 0 is ")
#println(round.(value.(B1.rmp.m[:x0][:,1]);digits=1))
#println()
#println("x0 at node 5 is")
#println(round.(value.(B3.rmp.m[:x0][:,6]);digits=1))
#println()
#println("x0 at node 14 is")
#println(round.(value.(B3.rmp.m[:x0][:,15]);digits=1))
#println()
#println("x0 at node 23 is")
#println(round.(value.(B3.rmp.m[:x0][:,23]);digits=1))
#println()
#println()

B_solved = deepcopy(B);
B3 = expand_subtree(B_solved,3,nodeA,nodeB);
Lf = JuMP.objective_value(B3.rmp.m);
println("Lower bound on tree 3 after fixing nodes is $(Lf)")

println("Impact on tree 3 of fixing nodes to deterministic value is $((Lf-Lt)/Lt)")

#=
if st == 0
    B_solved = deepcopy(B);
    B1 = expand_subtree(B_solved,1);
    S1 = deepcopy(S);

    B_solved = deepcopy(B);
    B2 = expand_subtree(B_solved,2);
    S2 = deepcopy(S);

    B_solved = deepcopy(B);
    B3 = expand_subtree(B_solved,3);
    S3 = deepcopy(S);

    println("Checking bounds for B1...")
    check_bounds(B1,S1)

    println("Checking bounds for B2...")
    check_bounds(B2,S2)

    println("Checking bounds for B3...")
    check_bounds(B3,S3)
elseif st == 1
    B_solved = deepcopy(B);
    B12 = expand_subtree(B_solved,12);
    S12 = deepcopy(S);

    B_solved = deepcopy(B);
    B13 = expand_subtree(B_solved,13);
    S13 = deepcopy(S);

    println("Checking bounds for B12...")
    check_bounds(B12,S12)

    println("Checking bounds for B13...")
    check_bounds(B13,S13)
end
=#
