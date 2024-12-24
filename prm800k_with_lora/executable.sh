#task_nums=(22) #(22 21)

###################################### run for shuffle label
ranks=(32)
epochs=(8)


for r in ${ranks[@]}
do
  for epoch in ${epochs[@]}
  do
    echo $r $epoch
    # output_dir_1="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/checkpoint/"
    # sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_test.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_train.json" $output_dir_1 False
    output_dir_1="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/checkpoint_phase2/"
    sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_test_phase2.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_train_phase2.json" $output_dir_1 False
    # output_dir_2="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/finalanswer_checkpoint_phase2/"
    # sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/finalanswer_downsampled_prm800k_test_phase2.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/finalanswer_downsampled_prm800k_train_phase2.json" $output_dir_2 True
  done
done

# sbatch run_lora.sh 4 5 

# for r in ${ranks[@]}
# do
#   for epoch in ${epochs[@]}
#   do
#     echo $r $epoch
#     # output_dir_1="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/checkpoint/"
#     # sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_test.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_train.json" $output_dir_1
#     output_dir_2="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/finalanswer_checkpoint_phase2/"
#     sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/finalanswer_downsampled_prm800k_test_phase2.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/finalanswer_downsampled_prm800k_train_phase2.json" $output_dir_2 True
#   done
# done









