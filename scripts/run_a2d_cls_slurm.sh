#!/bin/bash
#SBATCH --partition=gpu-S --gres=gpu:1 --constraint=12G
d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/dg_trials
source activate pytorch
python $2  --config configs/office-train-config_CDA.yaml --source ./txt/source_amazon_cls.txt --target ./txt/target_dslr_cls.txt --gpu $1
