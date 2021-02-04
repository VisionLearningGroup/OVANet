#!/bin/bash
#SBATCH --partition=gpu-L --gres=gpu:1 --constraint="titanxp"
d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/dg_trials
source activate pytorch
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dreal_univ.txt --target txt/target_dpainting_univ.txt --gpu $1 --multi 0.1
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dreal_univ.txt --target txt/target_dpainting_univ.txt --gpu $1 --multi 0.5
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dreal_univ.txt --target txt/target_dpainting_univ.txt --gpu $1 --multi 1.0

