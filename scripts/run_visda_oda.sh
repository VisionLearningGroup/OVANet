#!/bin/bash
python $2  --config configs/visda-train-config_ODA.yaml --source txt/source_list.txt --target txt/target_list.txt --gpu $1

