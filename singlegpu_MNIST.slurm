#!/bin/bash
#SBATCH --job-name=mnist_ddp_1gpu_batch32
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Number of GPUs to use (adjust as needed)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4    # Adjust based on CPU needs
#SBATCH --gres=gpu:1          # Number of GPUs to request (adjust as needed)
#SBATCH --time=6:00:00       # Adjust time as needed (HH:MM:SS)
#SBATCH --mem=12GB           # Adjust memory as needed
#SBATCH --account=bgross
#SBATCH --partition=gpu
#SBATCH --output=mnist_ddp_%j.out

postfix="mnist_ddp_1gpu_batch32"

# Load necessary modules
module load pytorch
module load nccl
module load cuda
module load cudnn

start_time=$(date +%s) # Record start time

# Run your PyTorch training script
time python singlegpu_MNIST.py 1000 100 32 >> out_$postfix.log # Adjust epochs, save_every and batch size as needed

end_time=$(date +%s) # Record end time
elapsed_seconds=$((end_time - start_time)) # Calculate elapsed time

# Convert seconds to hours, minutes, and seconds
elapsed_hours=$((elapsed_seconds / 3600))
elapsed_minutes=$(( (elapsed_seconds % 3600) / 60 ))
elapsed_seconds=$((elapsed_seconds % 60))

# Display and save the run time
echo "Total run time: ${elapsed_hours} hours, ${elapsed_minutes} minutes, ${elapsed_seconds} seconds"
echo "Total run time: ${elapsed_hours} hours, ${elapsed_minutes} minutes, ${elapsed_seconds} seconds" >> runtime_$postfix.log
