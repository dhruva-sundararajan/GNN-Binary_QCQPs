#!/bin/bash
#SBATCH -J data_generation_easy
#SBATCH --account=lgo
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=data_generation_easy.out
#SBATCH --error=data_generation_easy.err

# Activate your environment
source ~/.bashrc
module load Miniconda3
source activate gnn_xiong_env

# Run your GNN training script
python /home/dhruvasundar/all_feasible/codes/generate_data.py --type easy

# Deactivate your environment
conda deactivate