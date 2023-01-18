# kaggle-ps-s3e2

Kaggle Playground Series 3, Episode 3 competition

NOTES:

* Catboost by default seems to be the best.
* RandomForest performs slightly better than XGBoost.

Tasks to obtain the best model:

* [x] Basic eda
* [x] Cross-validation
* [x] First training
* [ ] Merge original dataset
* [ ] Implement LightGBM and Catboost
* [ ] Set order in smoke status
* [ ] Use AutoGluon framework
* [ ] Implement Lasso regression
* [ ] Use ten folds
* [ ] Try original dataset without adding label for original/synthetic
* [ ] Implement logistic regression
* [ ] Scale numerical variables between 0 and 1

## Train, validation & submission

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=lgbm]  # [rf|svd|xgb|lgbm|cb]
```

Submission is stored in outputs folder (see `config.py` for complete path)
