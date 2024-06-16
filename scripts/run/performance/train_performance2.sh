#!/bin/bash

#SBATCH --time=23:59:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --output=train-performance-2.%j.out

module purge

module load anaconda
conda activate dinov2

echo "== Starting! =="
sleep 3

cd /projects/skinder@xsede.org/repos/dinov2_cervical

sleep 3

torchrun dinov2/train/train.py \
    --rdzv-backend c10d \
    --rdzv-endpoint localhost:0 \
    --config-file /projects/skinder@xsede.org/repos/dinov2_cervical/dinov2/configs/train/vitl14_performance.yaml \
    --output-dir /scratch/alpine/skinder@xsede.org/cervix_dinov2/outputs/performance/run_2

echo "== End =="