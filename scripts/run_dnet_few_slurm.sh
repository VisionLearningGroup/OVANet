#!/bin/bash
#SBATCH --partition=gpu-M --gres=gpu:1 --constraint="12G"
d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/openet
source activate pytorch
python $2  --config configs/dnet-train-config_ssop.yaml --source txt/source_dreal125_open.txt --target_path txt/target_dpainting125_open_unl.txt  --target_label txt/target_dpainting125_open_few_5.txt --gpu $1

