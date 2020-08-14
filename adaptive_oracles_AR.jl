
using Distributions
using LinearAlgebra: dot
using JuMP
using Gurobi

#==================================================================
A simple implemention in Julia of the adaptive oracles Benders
decomposition algorithm for solving problems of the form
        min_x f(x) + ∑ π_i g(x_i, c_i)
where
    f(x) = α . x
and
    g(x, c) = min_y c . (C y)  such that Ay ≤ Bx and y ≧ 0

We assume that the matrices A, B and C have fixed entries in the interval
(0,1), that the vectors c_i have fixed entries in the interval (-1,0) and
that the decision variables x_i all satisfy 0 ≦ x_i ≦ 1

Usage:
=====

Generate a new pseduorandom problem instance with
julia> (PI,cs) = (n,c,d,x,y)
where
    • n is the number of subproblems,
    • c is the dimension of the cost vector c_i
    • d is the number of constraintts (ie the number of rows of A/B)
    • x is the dimension of the vector x_i
    • y is the dimension of the vector y

Then attempt to solve the problem to a tolerance of ϵ with
julia> benders(PI,cs,ϵ)

==================================================================#

const GUROBI_ENV = Gurobi.Env();

mutable struct HyperPlane{T}
     # represents a hyperplane of the form θ + λ . (x - y)
     y::Vector{T}
     θ::T
     λ::Vector{T}
end

locally_improving = function(hp1::HyperPlane{Float64},
                             hqs::Set{HyperPlane{Float64}})
    # hp1 is locally improving if and only if
    # hp1.θ > max (hq.θ + hq.λ . (hp.y - hq.q)) for all hq in hqs
    max_θ = -Inf;
    for hq in hqs
        max_θ = max(max_θ, hq.θ + dot(hq.λ, hp1.y - hq.y));
    end
    return (hp1.θ > max_θ)
end

mutable struct STuple{T}
    # respresents a tuple (x_i, c_i, θ_i, λ_i, ϕ_i)
    x_i::Vector{T}
    c_i::Vector{T}
    θ_i::T
    λ_i::Vector{T}
    ϕ_i::Vector{T}
end

mutable struct ProblemInstance
    I::Vector{Int64}
    π::Dict{Int64,Float64}
    α::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
end

generate = function(nodes::Int64,
                    cdim::Int64,
                    ddim::Int64,
                    xdim::Int64,
                    ydim::Int64)
    # for the given parameters, generate a random problem instance
    # together with a random vector c_i for each node i

    I = 1:nodes;

    π = Dict{Int64,Float64}();
    cs = Dict{Int64,Vector{Float64}}();

    α = Vector{Float64}(undef,nodes*xdim);
    A = Matrix{Float64}(undef,ddim,ydim);
    B = Matrix{Float64}(undef,ddim,xdim);
    C = Matrix{Float64}(undef,cdim,ydim);

    for (i,j) in Iterators.product(1:ddim,1:ydim)
        A[i,j] = Distributions.rand(Uniform(0,1));
    end

    for (i,j) in Iterators.product(1:ddim,1:xdim)
        B[i,j] = Distributions.rand(Uniform(0,1));
    end

    for (i,j) in Iterators.product(1:cdim,1:ydim)
        C[i,j] = Distributions.rand(Uniform(0,1));
    end

    α = Distributions.rand(Uniform(-10,-1),xdim*nodes);

    for j in I
        π[j] = Distributions.rand(Uniform(0,1));
        cs[j] = Distributions.rand(Uniform(-1,0),cdim);
    end

    return (ProblemInstance(I,π,α,A,B,C),cs)
end

g = function(x::Vector{Float64}, c::Vector{Float64}, PI::ProblemInstance)
    # for given vectors x and c solve the LP
    #        min c.Cy such that A * y ≦ B*x and y ≧ 0
    # and return the objective value θ as well as subgradients λ with
    # respect x and ϕ with respect to c

    if (length(x) != size(PI.B)[2] || size(PI.B)[1] != size(PI.A)[1] ||
    size(PI.A)[2] != size(PI.C)[2] || size(PI.C)[1] != length(c) )
        error("Dimensions of given problem data are inconsistent.")
    end

    sp = Model(optimizer_with_attributes(() ->
                  Gurobi.Optimizer(GUROBI_ENV)));
    set_silent(sp);

    (m,n) = size(PI.A);
    p = length(c);
    q = length(x);

    @variable(sp, y[1:n]);
    @constraint(sp, pos[i in 1:n], y[i] >= 0);
    @constraint(sp, con[j in 1:m], (PI.A*y)[j] <= (PI.B*x)[j]);
    @objective(sp, Min, dot(PI.C'*c,y));

    optimize!(sp);

    if termination_status(sp) != MOI.OPTIMAL
        println(sp)
        error("No optimal solution was found; something went wrong.")
    end

    θ = objective_value(sp);
    λ = PI.B' * dual.(con);
    ϕ = PI.C * value.(y);

    return (θ, λ, ϕ)
end

ao_lb = function(x::Vector{Float64},
                 c::Vector{Float64},
                 S::Array{STuple{Float64}})
    # given an array S of points (x_i, c_i, θ_i, λ_i, ϕ_i) calculate a
    # lower bound θ and a subgradient λ for the point g(x,c) by solving
    #       max θ = ∑  μ_s (θ_s + λ_s^T (x - x_s))
    #       such that μ_s ≧ 0 and
    #                 ∑ μ_s = 1 and
    #                 ∑ μ_s c_s ≦ c

    lp = Model(optimizer_with_attributes(() ->
                         Gurobi.Optimizer(GUROBI_ENV)));
    set_silent(lp);

    n = length(x);
    m = length(c);
    r = length(S);

    @variable(lp, μ[1:r]);

    @constraint(lp, pos[i in 1:r], μ[i] >= 0);
    @constraint(lp, unit_sum,
                    sum(μ[i] for i in 1:r) == 1);
    @constraint(lp, con[j in 1:m],
                    sum(μ[i]*S[i].c_i[j] for i in 1:r) <= c[j]);

    z = AffExpr(0);
    for s in 1:r
      add_to_expression!(z, μ[s]*(S[s].θ_i + dot(S[s].λ_i,x - S[s].x_i)));
    end

    @objective(lp, Max, z);

    optimize!(lp);

    if termination_status(lp) != MOI.OPTIMAL
        println(lp)
        error("No optimal solution was found; something went wrong.")
    end

    μ_star = value.(μ);

    return (objective_value(lp), sum(μ_star[s]*S[s].λ_i for s in 1:r))
end

ao_ub = function(x::Vector{Float64},
                 c::Vector{Float64},
                 S::Array{STuple{Float64}})
    # given an array S of points (x_i, c_i, θ_i, λ_i, ϕ_i) calculate an
    # upper bound θ for the point g(x,c) by solving
    #       min θ = ∑  μ_s (θ_s + ϕ_s^T (c - c_s))
    #       such that μ_s ≧ 0 and
    #                 ∑ μ_s = 1 and
    #                 ∑ μ_s x_s ≦ x

    lp = Model(optimizer_with_attributes(() ->
                         Gurobi.Optimizer(GUROBI_ENV)));
    set_silent(lp);

    n = length(x);
    m = length(c);
    r = length(S);

    @variable(lp, μ[1:r]);

    @constraint(lp, pos[i in 1:r], μ[i] >= 0);
    @constraint(lp, unit_sum,
                    sum(μ[i] for i in 1:r) == 1);
    @constraint(lp, con[j in 1:n],
                    sum(μ[i]*S[i].x_i[j] for i in 1:r) <= x[j]);

    z = AffExpr(0);
    for s in 1:r
        add_to_expression!(z, μ[s]*(S[s].θ_i + dot(S[s].ϕ_i,c - S[s].c_i)));
    end
    @objective(lp, Min, z);

    optimize!(lp);

    if termination_status(lp) != MOI.OPTIMAL
        println(lp)
        error("No optimal solution was found; something went wrong.")
    end

    return objective_value(lp)
end

i_ex1 = function(E::Vector{Int64},
                 π::Dict{Int64,Float64},
                 U::Dict{Int64,Float64},
                 L::Dict{Int64,Float64})
    # given weights π and values U and L for a collection
    # of nodes E, return (1) the element of i which maximises the
    # value of π_i (U_i - L_i) and (2) the rest of the set
    # as an ordered pair
    V = Dict{Int64,Float64}();
    for i in E
        V[i] = π[i]*(U[i] - L[i]);
    end

    i_ex = findmax(V)[2];
    rest = setdiff(E,i_ex);

    return (i_ex, rest)
end

lb = function(PI::ProblemInstance)
    # returns an a priori lower bound for β for a given problem instance
    (m,n) = size(PI.A);
    x = maximum(abs.(PI.C*inv(PI.A)*PI.B))
    return -m*n*x;
end

consult_exact_oracles = function(PI::ProblemInstance,
                                 x_star::Vector{Float64},
                                 cs::Dict{Int64,Vector{Float64}},
                                 Θ::Dict{Int64,Set{HyperPlane{Float64}}},
                                 i::Int64)
    n = size(PI.B)[2];

    x_i = x_star[((i-1)*n+1):i*n];
    c_i = cs[i];

    (θ, λ, ϕ) = g(x_i,c_i,PI);

    new_p = HyperPlane(x_i, θ, λ);
    new_S = STuple(x_i, c_i, θ, λ, ϕ);

    γ = locally_improving(new_p,Θ[i]);

    return (γ, new_p, new_S)
end

benders = function(PI::ProblemInstance,
                   cs::Dict{Int64,Vector{Float64}},
                   ϵ::Float64)
    # applies the adaptive-oracles version of the Benders decomposition
    # algorithm to minimize f(x) + ∑ π_i g(x,c)

    # sets up the relaxed master problem for the given problem instance
    # and initializes the set of hyperplanes Θ the set of solutions S

    if (size(PI.B)[1] != size(PI.A)[1] || size(PI.A)[2] != size(PI.C)[2])
        error("Dimensions of given problem data are inconsistent.")
    end

    (m,n) = size(PI.A);
    p = size(PI.C)[1];
    q = size(PI.B)[2];

    N = length(PI.I)*q;

    β_0 = lb(PI);

    rmp = Model(optimizer_with_attributes(() ->
                       Gurobi.Optimizer(GUROBI_ENV)));
    set_silent(rmp);

    @variable(rmp, x[1:N]);
    @variable(rmp, β[i in PI.I]);

    @constraint(rmp, x_feasible[j in 1:N], 0 <= x[j] <= 1);
    @constraint(rmp, β_feasible[i in PI.I], β[i] >= β_0);

    @objective(rmp, Min, sum(PI.α[i] * x[i] for i in 1:N) +
                         sum(PI.π[i]*β[i] for i in PI.I));

    x_0 = zeros(q);
    c_0 = -ones(q);
    (θ, λ, ϕ) = g(x_0,c_0,PI);

    S = [STuple(x_0,c_0,θ,λ,ϕ)];

    Θ = Dict{Int64,Set{HyperPlane{Float64}}}();
    for i in PI.I
        Θ[i] = Set{HyperPlane{Float64}}();
        push!(Θ[i],HyperPlane(x_0,β_0,λ));
    end

    UBs = Dict{Int64,Float64}();
    LBs = Dict{Int64,Float64}();
    hist = Dict{Int64,Int64}();

    for i in PI.I
        UBs[i] = Inf;
        LBs[i] = -Inf;
    end

    global (j,U,L) = (0, 999999999, -999999999);

    while (U-L)/abs(U) > ϵ
        global j = j+1;

        optimize!(rmp);

        if termination_status(rmp) != MOI.OPTIMAL
            println(rmp);
            error("No optimal solution was found; something went wrong.")
        end

        (x_star, β_star) = (value.(x), value.(β));

        global L = dot(PI.α, x_star) +
                      sum(PI.π[i]*β_star[i] for i in PI.I);
        global ξ = false;
        global E = PI.I;

        # solve exact oracles to find a locally improving hyperplane
        while !ξ && !isempty(E)
            global (i_ex,E) = i_ex1(E,PI.π,UBs,LBs);
            (γ, new_p, new_S) =
                consult_exact_oracles(PI, x_star, cs, Θ, i_ex);

            S = vcat(S,new_S);
            push!(Θ[i_ex],new_p);
            haskey(hist,i_ex) ? hist[i_ex] = hist[i_ex] + 1 : hist[i_ex] = 1;
            global ξ = γ;

            # update local lower and upper bounds and update RMP

            global LBs[i_ex] = new_S.θ_i;
            global UBs[i_ex] = new_S.θ_i;

            #= Only include for debugging...
            println("Exact oracles give ...")
            println("LBs[",i_ex,"] = ",LBs[i_ex]);
            println("LBs[",i_ex,"] = ",LBs[i_ex]);
            =#

            rng = ((i_ex-1)*q+1):i_ex*q;
            x_i = x_star[rng];

            expr0 = @expression(rmp, new_S.θ_i - dot(new_S.λ_i,x_i));
            expr1 = @expression(rmp,
                        sum(new_S.λ_i[k]*x[rng][k] for k in 1:q));
            @constraint(rmp, β[i_ex] >= expr0 + expr1)
        end;

        # use adaptive oracles for the remaining subproblems
        n = size(PI.B)[2];

        for i in E
            rng = ((i-1)*n+1):i*n;
            x_i = x_star[rng];
            c_i = cs[i];

            (lb,λ) = ao_lb(x_i,c_i,S);
            global LBs[i] = lb;

            ub = ao_ub(x_i,c_i,S);
            global UBs[i] = ub;

            #= Only include for debugging...
            println("Adaptive oracles give ...")
            println("LBs[",i,"] = ",LBs[i]);
            println("UBs[",i,"] = ",UBs[i]);
            =#

            push!(Θ[i],HyperPlane(x_i,lb,λ));

            expr0 = @expression(rmp, lb - dot(λ,x_i));
            expr1 = @expression(rmp,
                      sum(λ[k]*x[rng][k] for k in 1:q));
            @constraint(rmp, β[i] >= expr0 + expr1)
        end

        global U = min(U, dot(PI.α, x_star) +
                               sum(PI.π[i]*UBs[i] for i in PI.I));
        println("After iteration ",j,", L = ",L," and U =",U)
    end

    return (L,U,hist);
end
