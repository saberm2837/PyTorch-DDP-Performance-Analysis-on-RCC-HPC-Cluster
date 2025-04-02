#!/bin/bash
#SBATCH --job-name=big_mnist_1gpu_batch512
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --nodelist=gn01      # Explicitly request a node
#SBATCH --ntasks-per-node=1  # Number of GPUs to use (adjust as needed)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8    # Adjust based on CPU needs
##SBATCH --gres=gpu:1        # Number of GPUs to request (adjust as needed)
#SBATCH --time=24:00:00      # Adjust time as needed (HH:MM:SS)
#SBATCH --mem=32GB           # Adjust memory as needed
#SBATCH --account=bgross
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out

postfix="big_mnist_1gpu_batch512"

# Load necessary modules
module load pytorch
module load nccl
module load cuda
module load cudnn

start_time=$(date +%s) # Record start time

# Run your PyTorch training script
time python singlegpu_bigMNIST.py 10 10 512 >> out_$postfix.log # Adjust epochs, save_every and batch size as needed

end_time=$(date +%s) # Record end time
elapsed_seconds=$((end_time - start_time)) # Calculate elapsed time

# Convert seconds to hours, minutes, and seconds
elapsed_hours=$((elapsed_seconds / 3600))
elapsed_minutes=$(( (elapsed_seconds % 3600) / 60 ))
elapsed_seconds=$((elapsed_seconds % 60))

# Display and save the run time
echo "Total run time: ${elapsed_hours} hours, ${elapsed_minutes} minutes, ${elapsed_seconds} seconds"
echo "Total run time: ${elapsed_hours} hours, ${elapsed_minutes} minutes, ${elapsed_seconds} seconds" >> runtime_$postfix.log
