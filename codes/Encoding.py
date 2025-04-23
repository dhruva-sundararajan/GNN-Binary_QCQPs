# %%
import gurobipy as gp
import numpy as np
import pandas as pd
import argparse
import pickle
import os
from utils import gurobi_env as env

__vtype__ = {'C': [1, 0, 0],
              'B': [0, 1, 0],
              'I': [0, 0, 1]}

__ctype__ = {'<': [1, 0, 0],
              '>': [0, 1, 0],
              '=': [0, 0, 1]}

__obj__ = {1: [1, 0],
            -1: [0, 1]}

def parse_solution_with_objective(sol_path: str) -> tuple[float, dict]:
    var_dict = {}
    objective = 0.0
    with open(sol_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("# Objective value"):
                try:
                    objective = float(line.split('=')[1])
                except:
                    pass
            else:
                try:
                    var, val = line.split()
                    var_dict[var.strip()] = float(val.strip())
                except:
                    continue
    return objective, var_dict

def get_solution_values(path_to_model, sol_dict) -> tuple:
    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f"{path_to_model} file not found")
    model = gp.read(path_to_model, env)
    vars = model.getVars()
    sol_values = []
    for var in vars:
        if var.VarName in sol_dict:
            sol_values.append(sol_dict[var.VarName])
        else:
            sol_values.append(0)
            sol_dict[var.VarName] = 0
    var_types = np.array(model.getAttr("VType"))
    return sol_values, var_types

def file2graph_HG(path_to_file: str, graph_dir: str) -> tuple:
    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"{path_to_file} file not found")
    file = os.path.basename(path_to_file)
    file_base = file.rsplit('.', 1)[0]
    os.makedirs(graph_dir, exist_ok=True)
    if os.path.exists(os.path.join(graph_dir, f"{file_base}.pkl")):
        name, data = pickle.load(open(os.path.join(graph_dir, f"{file_base}.pkl"), "rb"))
        if len(data) == 7:
            return (name, data)

    model = gp.read(path_to_file, env)
    edge_features, edges = [], []
    constr_features, obj_features = [], []
    row = [var.VarName for var in model.getVars()] + ["1", "sqr"] + [c.ConstrName for c in model.getConstrs()] + \
          [qc.QCName for qc in model.getQConstrs()] + ["obj"]
    row_index = pd.Index(row)
    obj_index = row_index.get_loc("obj")
    one_index = row_index.get_loc("1")
    sqr_index = row_index.get_loc("sqr")

    var_types = np.array(model.getAttr("VType"))
    var_lb = np.array(model.getAttr("LB"))
    var_ub = np.array(model.getAttr("UB"))
    var_lb_inf = np.isinf(var_lb).astype(int)
    var_ub_inf = np.isinf(var_ub).astype(int)
    var_lb = np.where(np.isinf(var_lb), 0, var_lb)
    var_ub = np.where(np.isinf(var_ub), 0, var_ub)
    var_types = np.array([__vtype__[v] for v in var_types]).T

    for constr in model.getConstrs():
        b = constr.getAttr("RHS")
        sense = constr.getAttr("Sense")
        constr_features.append(__ctype__[sense] + [b])
        line_expr = model.getRow(constr)
        for i in range(line_expr.size()):
            var = line_expr.getVar(i)
            coeff = line_expr.getCoeff(i)
            constr_index = row_index.get_loc(constr.ConstrName)
            var_index = row_index.get_loc(var.VarName)
            edges.append([constr_index, var_index, one_index])
            edge_features.append(coeff)

    for Qconstr in model.getQConstrs():
        b = Qconstr.getAttr("QCRHS")
        sense = Qconstr.getAttr("QCSense")
        constr_features.append(__ctype__[sense] + [b])
        quad_expr = model.getQCRow(Qconstr)
        quad_le = quad_expr.getLinExpr()
        for i in range(quad_le.size()):
            var = quad_le.getVar(i)
            coeff = quad_le.getCoeff(i)
            constr_index = row_index.get_loc(Qconstr.QCName)
            var_index = row_index.get_loc(var.VarName)
            edges.append([constr_index, var_index, one_index])
            edge_features.append(coeff)
        for i in range(quad_expr.size()):
            var1 = quad_expr.getVar1(i)
            var2 = quad_expr.getVar2(i)
            coeff = quad_expr.getCoeff(i)
            var1_index = row_index.get_loc(var1.VarName)
            var2_index = row_index.get_loc(var2.VarName)
            qconstr_index = row_index.get_loc(Qconstr.QCName)
            edges.append([qconstr_index, var1_index, sqr_index if var1_index == var2_index else var2_index])
            edge_features.append(coeff)

    obj = model.getObjective()
    obj_features.append(__obj__[model.getAttr("ModelSense")])
    if isinstance(obj, gp.QuadExpr):
        obj_le = obj.getLinExpr()
        for i in range(obj_le.size()):
            var = obj_le.getVar(i)
            coeff = obj_le.getCoeff(i)
            var_index = row_index.get_loc(var.VarName)
            edges.append([obj_index, var_index, one_index])
            edge_features.append(coeff)
        for i in range(obj.size()):
            var1 = obj.getVar1(i)
            var2 = obj.getVar2(i)
            coeff = obj.getCoeff(i)
            var1_index = row_index.get_loc(var1.VarName)
            var2_index = row_index.get_loc(var2.VarName)
            edges.append([obj_index, var1_index, sqr_index if var1_index == var2_index else var2_index])
            edge_features.append(coeff)
    else:
        for i in range(obj.size()):
            var = obj.getVar(i)
            coeff = obj.getCoeff(i)
            var_index = row_index.get_loc(var.VarName)
            edges.append([obj_index, var_index, one_index])
            edge_features.append(coeff)

    constr_features = np.array(constr_features).reshape(-1, 4)
    obj_features = np.array(obj_features).reshape(-1, 2)
    var_features = np.vstack((var_types, var_lb, var_ub, var_lb_inf, var_ub_inf,
                              np.zeros((8, var_types.shape[1])), np.random.rand(1, var_types.shape[1]))).T
    constr_features = np.hstack((np.zeros((constr_features.shape[0], 9)), constr_features,
                                 np.zeros((constr_features.shape[0], 2)), np.random.rand(constr_features.shape[0], 1)))
    obj_features = np.hstack((np.zeros((obj_features.shape[0], 13)), obj_features,
                              np.random.rand(obj_features.shape[0], 1)))
    one_features, sqr_features = np.zeros((1, 16)), np.zeros((1, 16))
    one_features[0, -1], sqr_features[0, -1] = np.random.rand(), np.random.rand()
    one_features[0, 7] = 1
    sqr_features[0, 8] = 1
    edge_features = np.array(edge_features)
    edges = np.array(edges, dtype=int)

    with open(os.path.join(graph_dir, file.replace(".lp", ".pkl")), "wb") as f:
        pickle.dump((file_base, [var_features, one_features, sqr_features, constr_features,
                    obj_features, edge_features, edges]), f)
    return (file_base, [var_features, one_features, sqr_features, constr_features, obj_features, edge_features, edges])

def initialize_dir(difficulty: str) -> tuple:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sol_dir = os.path.join(_root, "data", "train", "problem", difficulty)
    input_dir = os.path.join(_root, "data",  "train", "problem", difficulty)
    output_dir = os.path.join(_root, "data", "train", "HG", "train", difficulty)
    graph_dir = os.path.join(_root, "data", "train", "HG", "graph", difficulty)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return (_root, sol_dir, input_dir, output_dir, graph_dir)

def graph_encoding(difficulty: str) -> None:
    _root, sol_dir, input_dir, output_dir, graph_dir = initialize_dir(difficulty)
    count = 0
    failures = {}
    for file in os.listdir(input_dir):
        if file.endswith(".lp"):
            try:
                file_base = file.rsplit('.', 1)[0]
                lp_path = os.path.join(input_dir, file)
                name, graph_data = file2graph_HG(lp_path, graph_dir)
                sol_subdir = os.path.join(sol_dir, file_base)
                if not os.path.isdir(sol_subdir):
                    failures[file] = "Solution subdir missing"
                    print(f"Missing solution directory for {file_base}")
                    continue
                y_list = []
                for sol_file in sorted(os.listdir(sol_subdir)):
                    if sol_file.endswith(".sol"):
                        sol_path = os.path.join(sol_subdir, sol_file)
                        obj_value, sol_dict = parse_solution_with_objective(sol_path)
                        sol_values, var_types = get_solution_values(lp_path, sol_dict)
                        y_vec = [obj_value] + sol_values
                        y_list.append(y_vec)
                if not y_list:
                    raise ValueError("No valid solutions found")
                y_matrix = np.array(y_list)
                graph_data.append(y_matrix)
                with open(os.path.join(output_dir, f"{file_base}.pkl"), "wb") as f:
                    pickle.dump((name, graph_data), f)
                print(f"Finished {file}")
                count += 1
            except Exception as e:
                failures[file] = e
                print(f"Failed {file}: {e}")
    print(f"Finished {count} {difficulty} files in total")
    print(f"Failed {len(failures)} {difficulty} files in total")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty', '-d', type=str, required=True, help='difficulty level')
    return parser.parse_args()

# %%
if __name__ == '__main__':
    args = parse_args()
    graph_encoding(**vars(args))