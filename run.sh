#!/usr/bin
set -e

cd src
conda activate ml
python -W ignore create_folds.py
python -W ignore train.py --model=cb
conda deactivate

conda activate autogluon
python -W ignore train_autogluon.py
conda deactivate
