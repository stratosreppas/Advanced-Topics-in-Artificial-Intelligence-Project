from gurobipy import Model, GRB, quicksum


def OCTSVM(x, y, D=2, c1=0.1, c2=0.1, c3=0.1, M=100000):
    
    def parent(t):
        return (t-1) // 2
    
    # Initialize model
    model = Model("OCTSVM")
    
    # Constants
    N = len(x)
    T = sum(2**D for i in range(D))
    p = len(x[0])
    
    
    # Variables
    delta = model.addVar(lb=0, name="delta")
    w = {(t, j): model.addVar(lb=-GRB.INFINITY, name=f"w_{t}") for t in range(T) for j in range(p)}
    # weight_0 = model.addVars(T, lb=-GRB.INFINITY, name="weight_0")
    e = {(i, t): model.addVar(lb=0, name=f"e_{i}_{t}") for t in range(T) for i in range(N)}
    beta = {(i, t, j): model.addVar(lb=-GRB.INFINITY, name=f"beta_{i}_{t}") for t in range(T) for i in range(N) for j in range(p)}
    z = {(i, t): model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t}") for t in range(T) for i in range(N)}
    theta = {(i, t): model.addVar(vtype=GRB.BINARY, name=f"theta_{i}_{t}") for t in range(T) for i in range(N)}
    xi = {(i, t): model.addVar(vtype=GRB.BINARY, name=f"xi_{i}_{t}") for t in range(T) for i in range(N)}
    d = {t: model.addVar(lb=0, name=f"d_{t}") for t in range(T)}

    # Objective function
    model.setObjective(delta + 
                       c1 * quicksum(e[i][t] for i in range(N) for t in range(T)) +
                       c2 * quicksum(xi[i][t] for i in range(N) for t in range(T)) +
                       c3 * quicksum(d[t] for t in range(T)), GRB.MINIMIZE)

    # Constraints
    for t in range(T):
        model.addConstr(
            quicksum(w[t][j] * w[t][j] for j in range(p)) <= 2 * delta, 
            name=f"nsvm_constraint_{t}"
            )
        
        if(t != 0):
            model.addConstr(
                d[t] <= d[parent(t)],
                name=f"d_sanity_constraint_{t}"
            )

    for i in range(N):
        for t in range(T):
            
            model.addConstr(
                y[i]*(w[t]*x[i]) - 2*y[i]*(beta[i][t]*x[i]) >= 1 - e[i][t] - M * (1 - z[i][t]),
                name=f"RE-SVM_constraint_{i}_{t}"
            )
            
            model.addConstr(
                quicksum(w[t][j] * w[t][j] for j in range(p)) <= M * d[t],
                name=f"omega_d_constraint_{t}"
            )
            
            for j in range(p):
                model.addConstr(
                    beta[i][t][j] == w[t][j]*xi[i][t],
                    name=f"beta_definition_constraint_{i}_{t}_{j}"
                )
                              
            model.addConstr(
                quicksum(w[t][j] * w[t][j] for j in range(p)) <= M * d[t],
                name=f"d_definition_constraint_{t}"
            )
            
            if(t != 0):
                model.addConstr(
                    z[i][t] <= z[i][parent(t)],
                    name=f"z_sanity_constraint_{i}_{t}"
                )
            
            model.addConstr(
                w[t]*x[i] >= -M * (1 - theta[i][t]),
                name=f"sanity_check_2_contraint_{i}_{t}"
            )
            
            model.addConstr(
                w[t]*x[i] <= M * theta[i][t],
                name=f"sanity_check_3_contraint_{i}_{t}"
            )
            
            if(t != 0):
                if t%2 == 1:
                    model.addConstr(
                        z[i][parent(t)] - z[i][t] <= theta[i][parent(t)],
                        name=f"left_inheritance_constraint_{i}_{t}"
                    )
                else:
                    model.addConstr(
                        z[i][parent(t)] -z[i][t]<= 1-theta[i][parent(t)],
                        name=f"right_inheritance_constraint_{i}_{t}"
                    )
            
        for d in range(D): 
            model.addConstr(
                quicksum(z[i][t] for t in range(2**d, 2**(d+1)) == 1),
                name = f"sanity_check_1_constraint_{i}"
            )


    # Solve the model
    model.optimize()

    # Display the results
    if model.status == GRB.OPTIMAL:
        print("Optimal objective value:", model.objVal)
        for v in model.getVars():
            print(f"{v.varName}: {v.x}")
