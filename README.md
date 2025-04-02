# PyTorch DDP Performance Analysis on RCC HPC Cluster

This repository contains PyTorch code and scripts used to analyze the runtime performance of Distributed Data Parallel (DDP) training across single GPU, multi-GPU (single node), and multi-node configurations on the RCC (Research Computing Center) HPC cluster. The primary goal is to understand the impact of GPU count and node distribution on training speed, particularly in environments with RoCEv2 interconnects and varying GPU hardware.

## Project Overview

The project focuses on benchmarking PyTorch DDP training using the MNIST dataset and a scaled-up version (bigMNIST) to stress GPU memory and communication. The experiments aim to:

* Compare training times for single GPU, multi-GPU (single node), and multi-node setups.
* Evaluate the impact of RoCEv2 interconnects on multi-node performance.
* Analyze the effect of varying GPU hardware configurations on training speed.
* Investigate the overhead of data loading and communication in distributed training.

## Dataset

* **MNIST:** The standard MNIST dataset (70,000 samples) is used for initial testing.
* **bigMNIST:** A larger dataset created by augmenting and replicating the MNIST dataset.
    * Augmentation: Random rotation, affine transformations, scaling, horizontal flips, and elastic deformations.
    * Replication: 6 augmented versions, then 200 copies of the combined dataset, resulting in 72,000,000 samples (~52GB).
    * Make sure that the dataset is big enough to not fit inside the available GPU memory

## Hardware Configuration (RCC HPC Cluster)

The RCC HPC cluster consists of three types of GPU nodes:

1.  **Nodes gn01-06:**
    * 48 cores
    * 360GB RAM
    * 4 x V100 GPUs (32GB GPU memory)
2.  **Nodes gn07-08:**
    * 48 cores
    * 480GB RAM
    * 4 x A40 GPUs (48GB GPU memory)
3.  **Node gn09:**
    * 40 cores
    * 512GB RAM
    * 4 x Tesla V100-SXM2 GPUs (32GB GPU memory)

* **Interconnect:** RoCEv2 (100 Gbps, 5us latency). Note: No InfiniBand or NVLink.

## Files

* singlegpu_MNIST.py            # Single GPU training script with MNIST dataset
* multigpu_MNIST.py             # Multi-GPU (single node) training script using DDP with MNIST dataset
* multinode_MNIST.py            # Multi-node training script using DDP with MNIST dataset
* singlegpu_MNIST.slurm         # SLURM script for single GPU training with MNIST dataset
* multigpu_MNIST.slurm          # SLURM script for multi-GPU training with MNIST dataset
* multinode_MNIST.slurm         # SLURM script for multi-node training with MNIST dataset

* singlegpu_bigMNIST.py         # Single GPU training script with custom-made bigMNIST dataset
* multigpu_bigMNIST.py          # Multi-GPU (single node) training script using DDP with custom-made bigMNIST dataset
* multinode_bigMNIST.py         # Multi-node training script using DDP with custom-made bigMNIST dataset
* singlegpu_bigMNIST.slurm      # SLURM script for single GPU training with custom-made bigMNIST dataset
* multigpu_bigMNIST.slurm       # SLURM script for multi-GPU training with custom-made bigMNIST dataset
* multinode_bigMNIST.slurm      # SLURM script for multi-node training with custom-made bigMNIST dataset

* README.md                    # This file
* requirements.txt             # Python dependencies
* results.md                   # Performance results

## Performance Results

The performance results for training on the MNIST and bigMNIST datasets are available in the **results.md**

**Summary of Key Findings:**

* Speedup is close to linear for multi-GPU training on a single node.
* Multi-node training is slower due to communication overhead.
* Performance varies significantly across different GPU nodes.

## Requirements

* Python 3.x
* PyTorch
* Torchvision

To install requirements:

```bash
pip install -r requirements.txt

