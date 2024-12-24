#!/bin/bash
#SBATCH --job-name=prm_gemma
#SBATCH --output=out/%A%a.out
#SBATCH --error=out/%A%a.err
#SBATCH --cpus-per-task=6 
# SBATCH --gres=gpu:a100:1                            
#SBATCH --gres=gpu:1
#SBATCH --mem=48G 
#SBATCH --time=47:50:00
#SBATCH --partition=main
# SBATCH --constraint='ampere'

module load python/3.10
module load cuda/12.1.1
source ../envs/veri_env/bin/activate 
HYDRA_FULL_ERROR=1

r=$1
epoch=$2
test_data=$3
train_data=$4
output_dir=$5
finalanswer=$6

echo $r $epoch $test_data $train_data
python main_4.py peft.r=$r training.num_train_epochs=$epoch dataset.test_file=$test_data dataset.train_file=$train_data training.output_dir=$output_dir finalanswer=$finalanswer







 
