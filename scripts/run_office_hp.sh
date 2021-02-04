#!/bin/bash
#SBATCH --partition=gpu-L --gres=gpu:1 --constraint="titanxp"

d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/dg_trials

source activate pytorch
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda.txt --gpu $1 --multi 0.1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda.txt --gpu $1 --multi 0.3
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda.txt --gpu $1 --multi 0.5


