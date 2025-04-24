import numpy as np
import pandas as pd
import time
import torch
import os
import gurobipy as gp
import argparse

import NeuralPrediction
from Encoding import file2graph_HG

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _encoding(input_file, data_dir):
    name, graph_data = file2graph_HG(input_file, data_dir)
    n_vars = len(graph_data[0])
    sol_values = np.zeros(n_vars)
    graph_data.append(sol_values)
    return name, graph_data

def predict(args, input_dir, input_file, data_dir):
    name, graph_data = _encoding(str(f"{input_dir}/{input_file}"), str(data_dir))
    var_features, _, _, constr_features, obj_features, edge_features, edges, _ = graph_data
    problem_data = [var_features, constr_features, obj_features, edge_features, edges]
    ModelClass = NeuralPrediction.all_models['UniEGNN']
    nnmodel = ModelClass(args)
    model_path = os.path.join(_root, 'models', args.difficulty)
    nnmodel.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
    G = NeuralPrediction.all_data["HG"](*graph_data)
    output = nnmodel(G)
    output = output[:len(G.opt_sol)].cpu().detach()
    
    output = torch.sigmoid(output)
    output = torch.where(output > args.threshold, torch.tensor(1.0),
                            torch.where(output < 1 - args.threshold, torch.tensor(0.0), output))
    output = output.cpu().detach().numpy()
    return output

def main(args):
    input_dir = os.path.join(_root, 'data', 'test', 'problem', args.difficulty)
    data_dir = os.path.join(_root, 'data', 'test', 'encoding', args.difficulty)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file = []
    no_gnn = []
    gnn = []
    variables_fixed = []

    for input_file in os.listdir(input_dir):
        if input_file.endswith('.lp'):
            start_time = time.time()
            model = gp.read(os.path.join(input_dir, input_file))
            model.optimize()
            end_time = time.time()
            time_taken_no_gnn = end_time - start_time
            start_time = time.time()
            output = predict(args, input_dir, input_file, data_dir)
            model = gp.read(os.path.join(input_dir, input_file))
            variables = model.getVars()
            v = 0
            for var, pred in zip(variables, predictions):
                if pred == 1.0 or pred == 0.0:
                    var.LB = pred
                    var.UB = pred
                    v += 1
            model.update()
            model.optimize()
            end_time = time.time()
            time_taken_with_gnn = end_time - start_time
            file.append(input_file)
            no_gnn.append(time_taken_no_gnn)
            gnn.append(time_taken_with_gnn)
            variables_fixed.append(v)
    df = pd.DataFrame({
        'File': file,
        'Time taken without GNN (s)': no_gnn,
        'Time taken with GNN (s)': gnn,
        'Variables fixed': variables_fixed
    })
    results_dir = os.path.join(_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, f'{args.difficulty}.csv'), index=False)
    
def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--threshold', '-t', type=float, default=0.9)
    args = parser.parse_args()
    args.bias = not args.nobias
    channels_dict = {
        'HG': (16, 1),
    }
    args.nfeat, args.nedge = channels_dict[args.encoding]

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)