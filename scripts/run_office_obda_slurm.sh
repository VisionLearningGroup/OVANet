#!/bin/bash
#SBATCH --partition=gpu-L --gres=gpu:1 --constraint=12G
d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/openet
source activate pytorch
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_dslr_obda.txt --gpu $1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda.txt --target ./txt/target_webcam_obda.txt --gpu $1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_webcam_obda.txt --target ./txt/target_amazon_obda.txt --gpu $1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_dslr_obda.txt --target ./txt/target_amazon_obda.txt --gpu $1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_dslr_obda.txt --target ./txt/target_webcam_obda.txt --gpu $1
python $2  --config configs/office-train-config_ODA.yaml --source ./txt/source_webcam_obda.txt --target ./txt/target_dslr_obda.txt --gpu $1