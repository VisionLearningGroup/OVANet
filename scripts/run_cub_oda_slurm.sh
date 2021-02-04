#!/bin/bash
#SBATCH --partition=gpu-M --gres=gpu:1 --constraint="12G"
d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/openet
source activate pytorch
python $2  --config configs/cub-train-config_ODA.yaml --source txt/cub_source.txt --target txt/cub_unl.txt --gpu $1
