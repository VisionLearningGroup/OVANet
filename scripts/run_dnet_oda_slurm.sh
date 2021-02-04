#!/bin/bash
#SBATCH --partition=gpu-M --gres=gpu:1 --constraint="titan"
d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/openet
source activate pytorch
python $2  --config configs/dnet-train-config_ODA.yaml --source ./txt/source_dreal125_open.txt --target ./txt/target_dpainting125_open.txt --gpu $1
python $2  --config configs/dnet-train-config_ODA.yaml --source ./txt/source_dclipart125_open.txt --target ./txt/target_dpainting125_open.txt --gpu $1
