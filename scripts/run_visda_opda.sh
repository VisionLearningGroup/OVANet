#!/bin/bash
python $2  --config configs/visda-train-config_UDA.yaml --source txt/source_list_univ.txt --target txt/target_list_univ.txt --gpu $1
