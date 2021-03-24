#!/bin/bash
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dreal_univ.txt --target txt/target_dpainting_univ.txt --gpu $1
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dreal_univ.txt --target txt/target_dsketch_univ.txt --gpu $1
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dsketch_univ.txt --target txt/target_dpainting_univ.txt --gpu $1
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dpainting_univ.txt --target txt/target_dreal_univ.txt --gpu $1
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dpainting_univ.txt --target txt/target_dsketch_univ.txt --gpu $1
python $2  --config configs/dnet-train-config_OPDA.yaml --source txt/source_dsketch_univ.txt --target txt/target_dreal_univ.txt --gpu $1

