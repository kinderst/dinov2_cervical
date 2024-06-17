#!/bin/bash

#SBATCH --time=23:59:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --output=train-research-b14.%j.out

module purge

module load anaconda
conda activate dinov2

echo "== Starting! =="
sleep 3

cd /projects/skinder@xsede.org/repos/dinov2_cervical

sleep 3

python -m torch.distributed.launch --master_port=25702 dinov2/train/train.py \
    --config-file /projects/skinder@xsede.org/repos/dinov2_cervical/dinov2/configs/train/vitb14_research_freeze.yaml \
    --output-dir /scratch/alpine/skinder@xsede.org/cervix_dinov2/outputs/research/run_vitb14_research_freeze

echo "== End =="