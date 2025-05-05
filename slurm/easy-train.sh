#!/bin/bash
#SBATCH -J easy-train
#SBATCH --account=lgo
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=easy-train.out
#SBATCH --error=easy-train.err

# Activate your environment
source ~/.bashrc
module load Miniconda3
source activate gnn_xiong_env

# Run your GNN training script
python /home/dhruvasundar/all_feasible/codes/Encoding.py -d easy
python /home/dhruvasundar/all_feasible/codes/NeuralPrediction.py -d easy

# Deactivate your environment
conda deactivate