#!/bin/bash
#SBATCH --partition=gpu-M --gres=gpu:1 --constraint="titanxp"

d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/openet
source activate pytorch
source=$3
target=$4
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_${source}_obda.txt --target ./txt/target_${target}_obda.txt --gpu $1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_${source}_obda.txt --target ./txt/target_${target}_obda_v2.txt --gpu $1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_${source}_obda.txt --target ./txt/target_${target}_obda_v3.txt --gpu $1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_${source}_obda.txt --target ./txt/target_${target}_obda_all.txt --gpu $1

