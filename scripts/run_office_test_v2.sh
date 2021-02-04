#!/bin/bash
#SBATCH --partition=gpu-M --gres=gpu:1

d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/openet
source activate pytorch
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda_all.txt --gpu $1 --close
python $3  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda_all.txt --gpu $1 --close
python $4  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda_all.txt --gpu $1 --close

