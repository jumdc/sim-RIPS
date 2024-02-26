#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH --job-name=clr
#SBATCH --output=/gpfswork/rech/oyr/urt67oj/out/simCLR.out
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1 
#SBATCH --gres=gpu:6 # reserver 1 GPU par noeud


echo "hello world"

module purge # nettoyer les modules herites par defaut

module load pytorch-gpu/py3/2.0.1
conda activate multimodal

python toyxp/train-sim.py \
paths=jz \
trainer=jz \
++prefix=simCLR \
++self_supervised.pretrained=True \
++self_supervised.epochs=200 \
++supervised.epochs=100 \
++supervised.lr=1e-4 \
++overfit_batches=0 \
++log=True \
++batch_size=1024 \
++num_workers=20 \
++logger.mode=offline \
++self_supervised.loss=vicREG