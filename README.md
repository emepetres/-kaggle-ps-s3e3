# kaggle-ps-s3e2

Kaggle Playground Series 3, Episode 3 competition

NOTES:

* Catboost by default seems to be the best.
* RandomForest performs slightly better than XGBoost.
* Interestingly, One Hot Encoding seems to work better than Label Enconding for lasso.

Tasks to obtain the best model:

* [x] Basic eda
* [x] Cross-validation
* [x] First training
* [x] Use AutoGluon framework -> improvement over catboost default
* [x] Use ten folds -> improvement by ~0.02%, but test public score disagree
* [x] Merge original dataset -> improvement on catboost validaton, and 0.4 points in autogluon public test score. catboost without feature for separating datasets, autogluon with it.
* [x] Implement Lasso regression -> better than xgboost, using one hot encoding
* [x] Implement logistic regression -> better than xgboost, using one hot encoding
* [x] Set "categorical_feature=" index as LightGBM parameter -> seems to work worse
* [ ] Ensemble of multiple algorithms
* [ ] Scale numerical variables between 0 and 1
* [ ] Hyperparameters tunning

## Train, validation & submission

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=lgbm]  # [rf|svd|xgb|lgbm|cb]
```

Submission is stored in outputs folder (see `config.py` for complete path)
