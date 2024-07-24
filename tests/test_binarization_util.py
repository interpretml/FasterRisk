import numpy as np
import pandas as pd
from fasterrisk.utils import isEqual_upTo_8decimal
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df, BinBinarizer

def import_pima_data_X_df():
    df = pd.read_csv("tests/pima_original_data.csv")
    X_df = df.drop(columns=['Outcome'])
    return X_df

def check_default_columnNames(X_binarized_df):
    rng = np.random.default_rng(seed=0)
    colnames = X_binarized_df.columns
    sampled_colnames = rng.choice(colnames, 5, replace=False)

    correct_answer = ['BMI<=34.2', 'Insulin<=245', 'BloodPressure<=94', 'SkinThickness<=18', 'DiabetesPedigreeFunction<=0.527']
    assert len(sampled_colnames) == len(correct_answer), "Number of sampled column names is not correct"
    assert np.all(sampled_colnames == correct_answer), "Column names are not correct"

def check_weighted_sampling_columnNames(X_binarized_df):
    rng = np.random.default_rng(seed=0)
    colnames = X_binarized_df.columns
    sampled_colnames = rng.choice(colnames, 5, replace=False)

    correct_answer = ['BMI<=31.1', 'Insulin<=237', 'BloodPressure<=94', 'SkinThickness<=18', 'DiabetesPedigreeFunction<=0.61']
    assert len(sampled_colnames) == len(correct_answer), "Number of sampled column names is not correct"
    assert np.all(sampled_colnames == correct_answer), "Column names are not correct"

def test_convert_continuous_df_to_binary_df_default():
    X_df = import_pima_data_X_df()

    X_binarized_df = convert_continuous_df_to_binary_df(X_df, max_num_thresholds_per_feature=100, sampling_weights='uniform', sampling_seed=0, get_featureIndex_to_groupIndex=False)

    check_default_columnNames(X_binarized_df)

def test_convert_continuous_df_to_binary_df_weighted_sampling():
    X_df = import_pima_data_X_df()

    X_binarized_df = convert_continuous_df_to_binary_df(X_df, max_num_thresholds_per_feature=100, sampling_weights='weighted', sampling_seed=0, get_featureIndex_to_groupIndex=False)

    check_weighted_sampling_columnNames(X_binarized_df)

def test_BinBinarizer_default():
    X_df = import_pima_data_X_df()

    binarizer = BinBinarizer(max_num_thresholds_per_feature=100, sampling_weights='uniform', sampling_seed=0)
    X_binarized_df, _ = binarizer.fit_transform(X_df)

    check_default_columnNames(X_binarized_df)

def test_BinBinarizer_weighted_sampling():
    X_df = import_pima_data_X_df()

    binarizer = BinBinarizer(max_num_thresholds_per_feature=100, sampling_weights='weighted', sampling_seed=0)
    X_binarized_df, _ = binarizer.fit_transform(X_df)

    check_weighted_sampling_columnNames(X_binarized_df)

if __name__ == '__main__':
    test_convert_continuous_df_to_binary_df_default()
    test_convert_continuous_df_to_binary_df_weighted_sampling()
    test_BinBinarizer_default()
    test_BinBinarizer_weighted_sampling()