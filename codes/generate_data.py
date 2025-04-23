import os
import argparse
import gurobipy as gp
from gurobipy import GRB

def solve_and_save_feasible_solutions(model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = gp.read(model_path)

    # Configure model to collect multiple feasible solutions
    model.setParam(GRB.Param.PoolSearchMode, 2)  # Find multiple feasible solutions
    model.setParam(GRB.Param.TimeLimit, 300)  # Set time limit to 600 secs
    model.optimize()

    # Check if feasible
    if model.SolCount == 0:
        print(f"No feasible solutions found for {model_path}.")
        return

    print(f"Found {model.SolCount} feasible solutions for {os.path.basename(model_path)}.")

    # Save each solution
    for i in range(model.SolCount):
        model.setParam(GRB.Param.SolutionNumber, i)
        sol_file = os.path.join(output_dir, f"solution_{i+1}.sol")
        model.write(sol_file)
        print(f"Saved: {sol_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Gurobi model and save feasible solutions.")
    parser.add_argument('--type', choices=['easy', 'medium'], required=True, help="Folder name under data/train/problem")

    args = parser.parse_args()

    train_path = os.path.join('/home/dhruvasundar/all_feasible/data/train/problem', args.type)

    for file in os.listdir(train_path):
        if file.endswith('.lp'):
            model_path = os.path.join(train_path, file)
            output_dir = os.path.splitext(model_path)[0]
            if os.path.isdir(output_dir) == False:
                solve_and_save_feasible_solutions(model_path, output_dir)

    test_path = os.path.join('/home/dhruvasundar/all_feasible/data/test/problem', args.type)

    for file in os.listdir(test_path):
        if file.endswith('.lp'):
            model_path = os.path.join(test_path, file)
            output_dir = os.path.splitext(model_path)[0]
            if os.path.isdir(output_dir) == False:
                solve_and_save_feasible_solutions(model_path, output_dir)