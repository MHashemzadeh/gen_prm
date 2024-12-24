#!/bin/bash
#SBATCH --job-name=prm_gemma
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=6 
#SBATCH --gres=gpu:a100:4                            
# SBATCH --gres=gpu:
#SBATCH --mem=48G 
#SBATCH --time=71:50:00
#SBATCH --partition=main
# SBATCH --constraint='ampere'

module load python/3.10
module load cuda/12.1.1
source ../envs/veri_env/bin/activate 
HYDRA_FULL_ERROR=1


python train_direct_prm.py config_direct.yaml






 
