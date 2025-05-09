import numpy as np
import pandas as pd # type: ignore
import time
import torch # type: ignore
import os
import gurobipy as gp # type: ignore
from gurobipy import GRB # type: ignore
import argparse
from random import uniform
import csv

import NeuralPrediction
from Encoding import file2graph_HG

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _encoding(input_file, data_dir):
    name, graph_data = file2graph_HG(input_file, data_dir)
    n_vars = len(graph_data[0])
    sol_values = np.zeros(n_vars)
    graph_data.append(sol_values)
    return name, graph_data

def sampling(args):
    input_dir = os.path.join(_root, 'data', 'test', args.difficulty)
    data_dir = os.path.join(_root, 'data', 'test', 'encoding', args.difficulty)
    results_dir = os.path.join(_root, 'results', args.solve_type, args.difficulty)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    instances = []
    for i in os.listdir(input_dir):
        if i.endswith('.lp'):
            instances.append(i)
    instances.sort()
    for i in os.listdir(input_dir):
        if i.endswith('.lp'):
            s = list()
            for _ in range(1000):
                model = gp.read(f"{input_dir}/{i}")
                variables = model.getVars()
                name, graph_data = _encoding(str(f"{input_dir}/{i}"), str(data_dir))
                var_features, _, _, constr_features, obj_features, edge_features, edges, _ = graph_data
                problem_data = [var_features, constr_features, obj_features, edge_features, edges]
                ModelClass = NeuralPrediction.all_models['UniEGNN']
                nnmodel = ModelClass(args)
                model_path = os.path.join(f"{args.difficulty}.pkl")
                nnmodel.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
                nnmodel.eval()
                G = NeuralPrediction.all_data["HG"](*graph_data)
                output = nnmodel(G)
                output = output[:len(G.opt_sol)].cpu().detach()
                predictions = torch.sigmoid(output)
                for var, pred in zip(variables, predictions):
                    num = uniform(0, 1)
                    if num <= pred:
                        var.LB = 1
                        var.UB = 1
                    else:
                        var.LB = 0
                        var.UB = 0
                model.update()
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL:
                    soln = {'status': model.Status, 'obj_val': model.ObjVal, 'vars': {}}
                    for var in model.getVars():
                        soln['vars'][var.VarName] = var.X
                else:
                    soln = {'status': model.Status, 'obj_val': None, 'vars': {}}
                    for var in model.getVars():
                        soln['vars'][var.VarName] = var.LB
                s.append(soln)
        rows = []
        for soln in s:
            row = {
                'status': soln['status'],
                'obj_val': soln['obj_val'],
            }
            for var in soln['vars']:
                row[var] = soln['vars'][var]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f"{results_dir}/{i.split('.')[0]}.csv", index=False)

def normal(args):
    input_dir = os.path.join(_root, 'data', 'test', 'problem', args.difficulty)
    data_dir = os.path.join(_root, 'data', 'test', 'encoding', args.difficulty)
    results_dir = os.path.join(_root, 'results', args.solve_type, args.difficulty)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    instances = []
    for i in os.listdir(input_dir):
        if i.endswith('.lp'):
            instances.append(i)
    instances.sort()
    log_path = os.path.join(results_dir, 'solution_status.csv')
    with open(log_path, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['Problem', 'Status', 'No. of variables fixed'])

        for i in instances:
            if i.endswith('.lp'):
                print(i)
                name, graph_data = _encoding(str(f"{input_dir}/{i}"), str(data_dir))
                var_features, _, _, constr_features, obj_features, edge_features, edges, _ = graph_data
                problem_data = [var_features, constr_features, obj_features, edge_features, edges]
                ModelClass = NeuralPrediction.all_models['UniEGNN']
                nnmodel = ModelClass(args)
                model_path = os.path.join(_root, 'models', f"{args.difficulty}.pkl")
                nnmodel.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
                nnmodel.eval()
                G = NeuralPrediction.all_data["HG"](*graph_data)
                output = nnmodel(G)
                output = output[:len(G.opt_sol)].cpu().detach()
                output = torch.sigmoid(output)
                predictions = torch.where(output > 0.9999, torch.tensor(1.0, device=output.device),
                    torch.where(output < 0.0001, torch.tensor(0.0, device=output.device), output))
                model = gp.read(os.path.join(input_dir, i))
                variables = model.getVars()
                v = 0
                for var, pred in zip(variables, predictions):
                    if pred == 0:
                        var.LB = 0
                        var.UB = 0
                        v += 1
                    elif pred == 1:
                        var.LB = 1
                        var.UB = 1
                        v += 1
                    else:
                        var.LB = 0
                        var.UB = 1
                model.update()
                model.optimize()

                base_name = os.path.splitext(i)[0]
                status = model.Status
                writer.writerow([i, status, v])

                if status == GRB.OPTIMAL:
                    sol_path = os.path.join(results_dir, f"{base_name}.sol")
                    model.write(sol_path)
                else:
                    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solve_type', '-s', type=str, required=True)
    parser.add_argument('--difficulty', '-d', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, default='UniEGNN',
                        choices=['UniEGNN'])
    parser.add_argument('--encoding', '-e', type=str, default='HG',
                        choices=['HG'])
    parser.add_argument('--nlayer', type=int, default=6)
    parser.add_argument('--nout', type=int, default=1, choices=[1])
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--input_drop', type=float, default=0.0)
    parser.add_argument('--first_aggregate', type=str, default='sum',
                        choices=['sum', 'mean', 'softmax_then_sum'])
    parser.add_argument('--second_aggregate', type=str,
                        default='mean', choices=['sum', 'mean', 'softmax_then_sum'])
    parser.add_argument('--nobias', action='store_true', default=False)
    parser.add_argument('--fix_edge', '-f', action='store_true', default=False)
    parser.add_argument('--activation', type=str, default='leakyrelu')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--vepoch', type=int, default=0,
                        help="0 for default epochs in one figure, -1 for default epochs in separate figures, integer for a specific epoch")
    parser.add_argument('--output', '-o', action='store_true',
                        default=False, help="output the neural outputs")
    parser.add_argument('--no-train', default=False)
    args = parser.parse_args()
    args.bias = not args.nobias
    channels_dict = {
        'HG': (16, 1),
    }
    args.nfeat, args.nedge = channels_dict[args.encoding]

    return args

if __name__ == '__main__':
    args = parse_args()
    eval(args.solve_type)(args)