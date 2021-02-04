#!/bin/bash
#SBATCH --partition=gpu-M --gres=gpu:1 --constraint="12G"
d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/openet
source activate pytorch
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dreal_univ.txt --target txt/target_dpainting_univ.txt --gpu $1
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dreal_univ.txt --target ./txt/target_dsketch_univ.txt --gpu $1
