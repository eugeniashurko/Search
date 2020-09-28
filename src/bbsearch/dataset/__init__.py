import csv
import pathlib

import pandas as pd


def norm_average_scores(columns, *, min=None, max=None):
    """Compute normalized average of columns.

    The values are averaged row-wise and normalized to the
    interval [min, max].

    If no values for `min` and/or `max` are provided then
    they will be determined based on the data.

    If only one column is provided then only the normalization
    is performed.

    Parameters
    ----------
    columns : pd.DataFrame or pd.Series
        The columns with the scores.
    min : int or float
    max : int or float

    Returns
    -------
    average : pd.Series
        The averaged scores.
    """
    if min is None:
        min = columns.values.min()
    if max is None:
        max = columns.values.max()
    average = (columns.mean(axis=1) - min) / (max - min)

    return average


def load_biosses(dataset_path):
    col_names = ["sentence_1", "sentence_2", "ann_1", "ann_2", "ann_3", "ann_4", "ann_5"]

    df = pd.read_csv(dataset_path, index_col=0, header=0, names=col_names)

    return df


def process_biosses(df_biosses):
    columns = {
        "ann_1": "score_1",
        "ann_2": "score_2",
        "ann_3": "score_3",
        "ann_4": "score_4",
        "ann_5": "score_5",
    }
    df_biosses["norm_score"] = norm_average_scores(df_biosses[columns], min=0, max=4)
    df_biosses = df_biosses.rename(columns=columns)
    df_biosses = df_biosses.reset_index(drop=True)

    return df_biosses


def load_ours(dataset_path):
    col_names = ["sentence_1", "sentence_2", "score_emmanuelle", "similarity"]

    df_ours = pd.read_csv(dataset_path, index_col=0, header=0, names=col_names, sep=";")

    return df_ours


def process_ours(df_ours):
    columns = {
        "score_emmanuelle": "score",
    }
    df_ours["norm_score"] = norm_average_scores(df_ours[columns], min=0, max=4)
    df_ours = df_ours.rename(columns=columns)
    df_ours = df_ours.reset_index(drop=True)

    return df_ours


def load_sts(dataset_path, split, name):
    dataset_path = pathlib.Path(dataset_path) / split

    df_data = pd.read_csv(
        dataset_path / f"STS.input.{name}.txt",
        sep="\t",
        header=None,
        names=["sentence_1", "sentence_2"],
        quoting=csv.QUOTE_NONE,
    )
    df_labels = pd.read_csv(
        dataset_path / f"STS.gs.{name}.txt",
        sep="\t",
        header=None,
        names=["score"]
    )

    return pd.concat([df_data, df_labels], axis=1)


def process_sts(ds_sts):
    ds_sts["norm_score"] = norm_average_scores(ds_sts[["score"]], min=0, max=5)

    return ds_sts


def load_all_sts(dataset_path):
    dataset_path = pathlib.Path(dataset_path)

    datasets_meta = {
        "train": {
            "MSRpar": "MSRpar",
            "MSRvid": "MSRvid",
            "SMTeuroparl": "SMTeuroparl",
        },
        "test-gold": {
            "MSRpar": "MSRpar",
            "MSRvid": "MSRvid",
            "SMTeuroparl": "SMTeuroparl",
            "SMTnews": "surprise.SMTnews",
            "OnWN": "surprise.OnWN"
        },
    }

    datasets = {}
    for split in datasets_meta:
        datasets[split] = {}

        for ds_name, ds_fname in datasets_meta[split].items():
            dataset = load_sts(dataset_path, split, ds_fname)
            datasets[split][ds_name] = process_sts(dataset)

    return datasets


def load_sick(dataset_path):
    dataset_path = pathlib.Path(dataset_path)
    df_sick = pd.read_csv(dataset_path / "SICK" / "SICK.txt", sep="\t")
    df_sick = df_sick.set_index("pair_ID")

    return df_sick


def process_sick(df_sick):
    columns = {
        "sentence_A": "sentence_1",
        "sentence_B": "sentence_2",
        "relatedness_score": "score",
    }
    df_sick = df_sick[columns]
    df_sick = df_sick.rename(columns=columns)
    df_sick["norm_score"] = norm_average_scores(df_sick[["score"]], min=1, max=5)
    df_sick = df_sick.reset_index(drop=True)

    return df_sick
