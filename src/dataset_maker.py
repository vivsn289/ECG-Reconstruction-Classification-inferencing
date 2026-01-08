# src/dataset_maker.py

import os
import ast
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    from src import config_pytorch as cfg
except:
    from src import config as cfg


def load_metadata():
    """Load ptbxl_database.csv + scp_statements.csv and build mapping."""
    meta_path = cfg.POSSIBLE_METADATA_PATHS[0]
    df = pd.read_csv(meta_path)

    scp_path = meta_path.replace("ptbxl_database.csv", "scp_statements.csv")
    scp_df = pd.read_csv(scp_path)

    # In your dataset: "Unnamed: 0" == scp_code
    scp_map = dict(zip(scp_df["Unnamed: 0"], scp_df["diagnostic_class"]))

    return df, scp_map


def parse_scp_codes(s):
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(s)
    except:
        return {}


def dominant_scp_to_class(scp_dict, scp_map):
    if not scp_dict:
        return None
    dominant = max(scp_dict, key=lambda k: scp_dict[k])
    return scp_map.get(dominant, None)


def process_and_split(seed=None, test_size=None, balance_train=False):
    seed = seed or cfg.RANDOM_SEED
    test_size = test_size or cfg.TEST_SIZE

    df, scp_map = load_metadata()

    df["scp_parsed"] = df["scp_codes"].apply(parse_scp_codes)
    df["diagnostic_class"] = df["scp_parsed"].apply(
        lambda d: dominant_scp_to_class(d, scp_map)
    )

    df = df[df["diagnostic_class"].notna()].reset_index(drop=True)

    print("\nLabel counts:")
    print(df["diagnostic_class"].value_counts())

    # Encode class
    label_encoder = {cls: i for i, cls in enumerate(cfg.CLASS_NAMES)}
    df["label"] = df["diagnostic_class"].map(label_encoder)

    # Patient-level split via strat_fold
    if "strat_fold" in df.columns:
        train_df = df[df["strat_fold"] != 10].copy()
        val_df = df[df["strat_fold"] == 10].copy()
    else:
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=seed)
        tr, va = next(splitter.split(df, groups=df["patient_id"]))
        train_df, val_df = df.iloc[tr].copy(), df.iloc[va].copy()

    print("\nTrain:", len(train_df))
    print("Val:", len(val_df))
    print(train_df["diagnostic_class"].value_counts())
    print(val_df["diagnostic_class"].value_counts())

    return train_df, val_df, label_encoder
