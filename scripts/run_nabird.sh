#!/bin/bash
python $2  --config configs/nabird-train-config_ODA.yaml --source txt/nabird_source.txt --target txt/nabird_unl.txt --gpu $1
