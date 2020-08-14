using Gurobi, JuMP, LinearAlgebra;

g = function(x)
	2*x[1]^2 - x[1]*x[2] + x[2]^2
    end;

s = function(x)
	[4*x[1] - x[2]; 
	 2*x[2] - x[1]]
    end;	

model = Model();

set_optimizer(model, Gurobi.Optimizer);
set_silent(model);

@variable(model, -1 <= y[1:2] <= 1);
@variable(model, beta);
@objective(model, Min, beta);

for k=0:20 
   if k == 0 
	global y_star = [-1.0; -1.0];
	global UB = g(y_star); 
   end;

   println("y = ", y_star);
 
   global UB = min(UB, g(y_star)); println("g = ", UB);

   grad = s(y_star); println("s = ", grad);
   supp = @expression(model, g(y_star) - dot(grad,y_star) + grad[1]*y[1] + grad[2]*y[2]  );
   @constraint(model, beta >= supp); println("beta >= ", supp);
   
   optimize!(model);
 
   global beta_star = value(beta); println("beta = ", beta_star);
   global y_star = value.(y); println("y_star = ", y_star);
       	
   k = k + 1;
   print("Stage k = ", k, "\n  Lower bound = ", beta_star, "\n  Upper bound =", UB,"\n\n")
end