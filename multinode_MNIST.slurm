#!/bin/bash
#SBATCH --job-name=mnist_ddp_4node_1gpu_batch32
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=12GB
#SBATCH --account=bgross
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out

postfix="mnist_ddp_4node_1gpu_batch32"

module purge
module load slurm
module load pytorch
module load nccl
module load cuda
module load cudnn

start_time=$(date +%s) # Record start time

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun -w "$head_node" --ntasks=1 hostname -i | awk '{print $1}')

echo "Head Node IP: $head_node_ip"
export LOGLEVEL=INFO

time \
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
torchrun \
--nnodes=4 \
--nproc_per_node=1 \
--rdzv_id=$SLURM_JOB_ID \
--rdzv_backend=c10d \
--rdzv_endpoint="$head_node_ip:29500" \
multinode_MNIST.py 1000 100 32 >> out_$postfix.log

end_time=$(date +%s) # Record end time
elapsed_seconds=$((end_time - start_time)) # Calculate elapsed time

# Convert seconds to hours, minutes, and seconds
elapsed_hours=$((elapsed_seconds / 3600))
elapsed_minutes=$(( (elapsed_seconds % 3600) / 60 ))
elapsed_seconds=$((elapsed_seconds % 60))

# Display and save the run time
echo "Total run time: ${elapsed_hours} hours, ${elapsed_minutes} minutes, ${elapsed_seconds} seconds"
echo "Total run time: ${elapsed_hours} hours, ${elapsed_minutes} minutes, ${elapsed_seconds} seconds" >> runtime_$postfix.log
