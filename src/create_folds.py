from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn import model_selection

from common.utils import set_seed
from common.kaggle import download_competition_data
import config


def _merge_with_original_data(
    data: pd.DataFrame,
    test: pd.DataFrame,
    add_origin_feature: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from common.kaggle import download_dataset

    original_path = Path(config.INPUTS) / "original"
    download_dataset(
        "pavansubhasht", "ibm-hr-analytics-attrition-dataset", original_path
    )

    original_data = pd.read_csv(
        original_path / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ).drop(columns="EmployeeNumber")
    synthetic_data = data.drop(columns="id")
    test_data = test.drop(columns="id")

    if add_origin_feature:
        original_data["synthetic_data"] = 0
        synthetic_data["synthetic_data"] = 1
        test_data["synthetic_data"] = 1

    merged_data = pd.concat([synthetic_data, original_data]).reset_index(drop=True)

    return merged_data, test_data


if __name__ == "__main__":
    set_seed(config.SEED)

    # Download data if necessary
    download_competition_data(config.COMPETITION, config.INPUTS)

    # Read training data
    df = pd.read_csv(config.TRAIN_DATA)
    df_test = pd.read_csv(config.TEST_DATA)

    # # df, df_test = _merge_with_original_data(df, df_test)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(
        n_splits=config.FOLDS, random_state=config.SEED, shuffle=True
    )

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df[config.TARGET].values)):
        df.loc[v_, "kfold"] = f

    # save the new csv with kfold column
    Path(config.TRAIN_FOLDS).parent.mkdir(exist_ok=True)
    df.to_csv(config.TRAIN_FOLDS, index=False)
    df_test.to_csv(config.PREPROCESSED_TEST_DATA, index=False)
