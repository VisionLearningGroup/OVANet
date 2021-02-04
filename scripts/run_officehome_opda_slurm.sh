#!/bin/bash
#SBATCH --partition=gpu-L --gres=gpu:1 --constraint=12G
d=$(date)
echo $d nvidia-smi
nvidia-smi
hostn=$(hostname -s)
cd /home/grad3/keisaito/domain_adaptation/openet
source activate pytorch
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Real_univ.txt --target ./txt/target_Art_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Real_univ.txt --target ./txt/target_Clipart_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Real_univ.txt --target ./txt/target_Product_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Product_univ.txt --target ./txt/target_Art_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Product_univ.txt --target ./txt/target_Clipart_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Product_univ.txt --target ./txt/target_Real_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Clipart_univ.txt --target ./txt/target_Real_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Clipart_univ.txt --target ./txt/target_Art_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Clipart_univ.txt --target ./txt/target_Product_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Art_univ.txt --target ./txt/target_Product_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Art_univ.txt --target ./txt/target_Real_univ.txt --gpu $1
python $2  --config configs/officehome-train-config_OPDA.yaml --source ./txt/source_Art_univ.txt --target ./txt/target_Clipart_univ.txt --gpu $1
