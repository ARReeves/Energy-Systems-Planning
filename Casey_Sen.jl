
using JuMP,Gurobi
using SymPy

include("Scenario_Trees.jl")

# code for generating/visualising directed trees/quivers

#================================================================
Code to solve three stage stochastic programming problems
of the same form as given in Section 6 of the Casey & Sen paper

That is, we assume we have vectors (of some dimension) x, y
and z and independent uniform random variables ξ_2 and ξ_3 and
that we wish to minimize

z = c_x' * x + c_y' * Ey(ξ_2) + c_z' * Ez(ξ_2, ξ_3)

such that

A_{x,x}'' x = b_x

A_{y,x}'' x + A_{y,y}'' y(ξ_2) = b_y(ξ_2)

A_{z,x}'' x + A_{z,y}'' y(ξ_2) + A_{z,z}'' y(ξ_2, ξ_3) = b_z(ξ_2, ξ_3)

x, y(ξ_2), z(ξ_2, ξ_3) ≧ 0

We do this by repeatedly splitting nodes of a scenario tree
T, starting with the initial degenerate tree 1 -> 2 -> 3, by
solving an aggregated linear problem corresponding to this
scenario tree and then exploring the nodal relaxations for
possibile infeasibilities to decide where to split next.

See the paper by Casey and Sen for details
(https://doi.org/10.1287/moor.1050.0146)
================================================================#

function initial_tree(ξ_2_lower,ξ_2_upper,ξ_3_lower,ξ_3_upper)
    # returns the degenerate scenario tree for given ξ_2, ξ_3
    return ScenarioTree(
        Quiver(                          # initial tree is 1 -> 2 -> 3
            1 => Node(1, EMPTY_SET, Set([2])),
            2 => Node(2, Set([1]), Set([3])),
            3 => Node(3, Set([2]), EMPTY_SET)
        ),
        Dict{Int64,Float64}(             # initial probabilites are all 1
            1 => 1.0,
            2 => 1.0,
            3 => 1.0
        ),
        Dict{Int64,Int64}(               # one node of each stage
            1 => 1,
            2 => 2,
            3 => 3
        ),
        Dict{Int64,Float64}(             # upper bounds as specified
            2 => ξ_2_upper,
            3 => ξ_3_upper
        ),
        Dict{Int64,Float64}(             # and lower bounds as specified
            2 => ξ_2_lower,
            3 => ξ_3_lower
        )
    )
end

function original_example()
    A = Matrix{Float64}([1 0 0; 1 0 0; 1 1 0; -1 -1 0; 0 1 1; 0 -1 01]);
    c = Vector{Float64}([5;0;12;12;10;10]);
    return (A,c)
end

# (ξ_2, ξ_3)  = symbols("ξ_2 ξ_3")

const GRB_ENV = Gurobi.Env();
setparam!(GRB_ENV,"Method",1);

# the three functions below adapated from Nicolo Mazzi's
# example_CaseySen.jl (with some very minor adjustments)

function cbasismap(cb::Symbol)
    (cb==:Basic) ? ic = true : ic = false
    return ic
end

function rbasismap(rb::Symbol)
    (rb==:Basic) ? ir = false : ir = true
    return ir
end

function get_basis_mtx(model::JuMP.Model)
    vlist = JuMP.all_variables(model);
    vint  = collect(1:length(vlist));

    gmodel = model.moi_backend.optimizer.model.inner;
    (cbasis,rbasis) = Gurobi.get_basis(gmodel);
    Amtx = Matrix(Gurobi.get_constrmatrix(gmodel));

    (vbool,cbool) = cbasismap.(cbasis),rbasismap.(rbasis);
    (vbasic,vnbasic) = (vint[vbool],vint[.!vbool]);
    Bmtx = Amtx[cbool,vbool];

    return (Bmtx,vlist,vbasic,vnbasic)
end

function agg_lp(A::Matrix{Float64},
                nx::Int64, ny::Int64, nz::Int64,
                c::Vector{Float64},
                T::ScenarioTree)

   # for a given scenario tree, formulate and solve the corresponding
   # aggregated linear programming problem

   rngx = 1:nx;
   rngy = (nx+1):(nx+ny);
   rngz = (nx+ny+1):(nx+ny+nz);

   s2 = sort(collect(filter(λ -> T.stages[λ] == 2, keys(T.stages))));
   s3 = sort(collect(filter(λ -> T.stages[λ] == 3, keys(T.stages))));

   t2 = length(s2);
   t3 = length(s3);

   A_xx = A[rngx,1];
   A_yx = A[rngy,1];
   A_zx = A[rngz,1];
   A_yy = A[rngy,2];
   A_zy = A[rngz,2];
   A_zz = A[rngz,3];

   lp = Model(optimizer_with_attributes(() ->
                 Gurobi.Optimizer(GRB_ENV, OutputFlag = 0)));

   @variable(lp,x[1:nx]);
   @variable(lp,y[1:ny*t2]);
   @variable(lp,z[1:nz*t3]);

   @constraint(lp, xpos[i in 1:nx], x[i] >= 0);
   @constraint(lp, ypos[j in 1:ny*t2], y[j] >= 0);
   @constraint(lp, zpos[k in 1:nz*t3], z[k] >= 0);

   @constraint(lp, con1, sum(A_xx[i]*x[i] for i in 1:nx) == 1);

   @constraint(lp, con2[n in 1:t2],
                   sum(A_yx[i]*x[i] for i in 1:nx) +
                   sum(A_yy[j]*y[(n-1)*ny+j] for j in 1:ny) ==
                   0.5*(T.upper[s2[n]] + T.lower[s2[n]]));

   @constraint(lp, con3[n in 1:t3],
                   sum(A_zx[i]*x[i] for i in 1:nx) +
                   sum(sum(A_zy[j]*
                        y[(findlast(λ -> s2[λ] == q,1:t2)-1)*ny+j]
                            for j in 1:ny)
                        for q in T.tree[s3[n]].sources) +
                   sum(A_zz[k]*z[(n-1)*nz+k] for k in 1:nz) ==
                   0.5*(T.upper[s3[n]] + T.lower[s3[n]]));

   w = AffExpr(0);
   for i in 1:nx
       add_to_expression!(w, c[i]*x[i]);
   end
   for n in 1:t2
       for j in 1:ny
           add_to_expression!(w, T.p[s2[n]]*c[j+nx]*y[(n-1)*ny+j]);
       end
   end
   for n in 1:t3
       for k in 1:nz
           add_to_expression!(w, T.p[s3[n]]*c[k+nx+ny]*z[(n-1)*nz+k]);
       end
   end

   @objective(lp, Min, w);

   # Debugging tool only; comment this out when done
   # println(lp)

   optimize!(lp);

   if termination_status(lp) != MOI.OPTIMAL
       error("No optimal solution was found; something went wrong.")
   end

   θ = objective_value(lp);

   x_star = value.(x);
   y_star = value.(y);
   z_star = value.(z);

   λ_x = Vector([dual(con1)]);
   λ_y = dual.(con2);
   λ_z = dual.(con3);

   #= Verbose: only for debugging
   println("This aggregated LP has optimal value ",θ)
   println("This is realised at x = ",x_star)
   println("y = ",y_star)
   println("z = ",z_star)
   println("The dual variable corresponding to the stage 1 node is ",
            dual(con1));
   println("The dual variables corresponding to the stage 2 nodes are ",
            dual.(con2));
   println("The dual variables corresponding to the stage 2 nodes are ",
            dual.(con3));
   =#

   return (θ , x_star , y_star , z_star , λ_x , λ_y , λ_z)

end


function nr(x_star::Vector{Float64},
            y_star::Vector{Float64},
            z_star::Vector{Float64},
               λ_x::Vector{Float64},
               λ_y::Vector{Float64},
               λ_z::Vector{Float64},
                 A::Matrix{Float64},
                 c::Vector{Float64},
                 T::ScenarioTree,
                 n::Node)
  # solves the nodal relaxation at node n,

  if !(T.stages[n.label] in Set([2 3]))
    error("Nodal relaxation only defined for stage 2 and stage 3 nodes.")
  end

  nx = length(x_star);                  # note λ_x is always of length 1
  ny = length(y_star) ÷ length(λ_y);
  nz = length(z_star) ÷ length(λ_z);

  rngx = 1:nx;
  rngy = (nx+1):(nx+ny);
  rngz = (nx+ny+1):(nx+ny+nz);

  A_xx = A[rngx,1];
  A_yx = A[rngy,1];
  A_zx = A[rngz,1];
  A_yy = A[rngy,2];
  A_zy = A[rngz,2];
  A_zz = A[rngz,3];

  s3 = sort(collect(filter(λ -> T.stages[λ] == 3, keys(T.stages))));
  t3 = length(s3);

  nr = Model(optimizer_with_attributes(() ->
                Gurobi.Optimizer(GRB_ENV, OutputFlag = 0)));

  if T.stages[n.label] == 2
    @variable(nr, u[1:ny]);

    @constraint(nr, ypos[j in 1:ny], u[j] >= 0);

    @constraint(nr, ncon, sum(A_yy[j]*u[j] for j in 1:ny) ==
                           0.5*(T.upper[n.label] + T.lower[n.label]) -
                            sum(A_yx[i]*x_star[i] for i in 1:nx));

    E_λ = 0.0;
    for i in 1:t3
        if (T.tree[s3[i]].label in n.targets)
            E_λ = E_λ + (T.p[s3[i]] / T.p[n.label]) * λ_z[i];
        end
    end

    @objective(nr, Min, sum(((c[j+nx]-E_λ*A_zy[j])*u[j]) for j in 1:ny));

else        # that if if T.stages[n.label] == 3
    @variable(nr, u[1:nz]);

    @constraint(nr, zpos[k in 1:nz], u[k] >= 0);

    @constraint(nr, ncon, sum(A_zz[k]*u[k] for k in 1:nz) ==
                           0.5*(T.upper[n.label] + T.lower[n.label]) -
                            sum(A_zx[i]*x_star[i] for i in 1:nx) -
                             sum(A_zy[j]*y_star[j] for j in 1:ny));

    @objective(nr, Min, sum(c[k+nx+ny]*u[k] for k in 1:nz));

  end

  # Debug only
  # println(nr)

  optimize!(nr)

  get_basis_mtx(nr)
  # return value.(u)

end

## currently the nr function only prints the nodal relaxation
#  problem - we actually want to solve this and to return an affine
# function of symbols ξ_2 and ξ_3 which we can use to check for
# infeasibility

# also I think on reflection that the dimensions of x,y and z must
# always be the same, as written -- should rewrite to reflect this
