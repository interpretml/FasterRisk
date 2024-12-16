from typing import *

import numpy as np
import pandas as pd
from sklearn.preprocessing._encoders import _BaseEncoder


def convert_continuous_df_to_binary_df(df, max_num_thresholds_per_feature=100, sampling_weights='uniform', sampling_seed=0, get_featureIndex_to_groupIndex=False):
    """Convert a dataframe with continuous features to a dataframe with binary features by thresholding

    Parameters
    ----------
    df : pandas.DataFrame
        original dataframe where there are columns with continuous features
    max_num_thresholds_per_feature : int, optional
        number of points we pick as thresholds if a column has too many unique values, by default 100
    sampling_weights : str, optional
        how to sample the thresholds from all unique values, by default 'uniform'; alternatively, 'weighted' allows to sample the thresholds according to the distribution of the unique values
    sampling_seed : int, optional
        random seed for sampling, by default 0
    get_featureIndex_to_groupIndex : bool, optional
        whether to return a numpy array that maps feature index to group index, by default False

    Returns
    -------
    binarized_df : pandas.DataFrame
        a new dataframe where each column only has 0/1 as the feature
    """

    colnames = df.columns
    n = len(df)
    rng = np.random.default_rng(seed=sampling_seed)
    print("Converting continuous features to binary features in the dataframe......")
    print("We select thresholds for each continuous feature by sampling (without replacement) <= max_num_thresholds_per_feature values from all unique values in that feature column.")

    binarized_dict = {}

    featureIndex_to_groupIndex = []
    for i in range(0, len(colnames)):
        # Step 1. If nan exists, get two new temporary columns, columnA and columnB;
            # columnA is obtained by dropping all nan values in the original column
            # columnB is obtained by replacing nan values with 0 in the original column
            # Moreover, create a new binary feature column for nan values in the original column
            # Set the temporary number of to-be-sampled thresholds by subtracting 1
        # If nan does not exist, both columnA and columnB are the original column
            # set the temporary number of to-be-sampled thresholds to the original number

        # Step 2
        # If #(unique values in columnA) <= 1, skip this column by executing 'continue'
        # If #(unique values in columnA) < temporary number of thresholds, set the temporary number of thresholds to #(unique values in columnA)
        # Sample temporary number of thresholds from unique values in columnA

        # Step 3
        # for each sampled threshold, create a new binary feature column

        # Step 1
        original_column = df[colnames[i]]
        tmp_num_thresholds = max_num_thresholds_per_feature 
        columnA = original_column.copy()
        columnB = original_column.copy()
        nan_indices = original_column.isnull()
        if nan_indices.sum() > 0:
            columnA = original_column.dropna()
            columnB = original_column.fillna(0)
            nan_feature = np.zeros(n)
            nan_feature[nan_indices] = 1
            binarized_dict[colnames[i] + "_isNaN"] = nan_feature
            tmp_num_thresholds -= 1
            featureIndex_to_groupIndex.append(i)

        # Step 2
        uni_columnA = columnA.unique()
        uni_columnA_counts = columnA.value_counts()

        if len(uni_columnA) <= 1:
            continue
        if len(uni_columnA) < tmp_num_thresholds:
            tmp_num_thresholds = len(uni_columnA)
        p = np.ones(len(uni_columnA)) / len(uni_columnA)
        if sampling_weights == 'weighted':
            p = uni_columnA_counts / uni_columnA_counts.sum()
        elif sampling_weights != 'uniform':
            raise ValueError("sampling_weights must be either 'uniform' or 'weighted'")
        sampled_indices = rng.choice(len(uni_columnA), tmp_num_thresholds, replace=False, p=p)
        sampled_thresholds = uni_columnA[sampled_indices]
        sampled_thresholds.sort()

        # Step 3
        for tmp_threshold in sampled_thresholds[:-1]:
            tmp_feature = np.ones(n, dtype=int)
            tmp_name = colnames[i] + "<=" + str(tmp_threshold)

            zero_indices = columnB > tmp_threshold
            tmp_feature[zero_indices] = 0

            binarized_dict[tmp_name] = tmp_feature
            featureIndex_to_groupIndex.append(i)

    binarized_df = pd.DataFrame(binarized_dict)
    print("Finish converting continuous features to binary features......")
    if get_featureIndex_to_groupIndex:
        return binarized_df, np.asarray(featureIndex_to_groupIndex, dtype=int)
    return binarized_df

def nan_onehot_single_column(column:pd.Series) -> np.ndarray:
    onehot = np.zeros(len(column))
    onehot[column.isnull()] = 1
    
    return onehot

class BinBinarizer(_BaseEncoder):
    """
    Binarize variables into binary variables based on percentile or user defined thresholds.
    
    Parameters
    ----------
    interval_width : int
        width of the interval measured by percentiles. For instance, if interval_width=10, then
        each interval will be between nth and (n+10)th percentile
    categorical_cols : list
        list of names for categorical variables
    wheter_interval : bool
        whether to one hot based on intervals or based on less thans, by default False (use less thans)
    """
    def __init__(self, whether_interval: bool=False, max_num_thresholds_per_feature: int=100, sampling_weights: str='uniform', sampling_seed: int=0, group_sparsity: bool=False) -> None:
        self.whether_interval = whether_interval
        self.group_sparsity = group_sparsity
        self.max_num_thresholds_per_feature = max_num_thresholds_per_feature
        self.sampling_weights = sampling_weights
        self.sampling_seed = sampling_seed
        self.rng = np.random.default_rng(seed=sampling_seed)
    
    def fit(self, df: pd.DataFrame) -> None:
        '''fit IntervalBinarizer'''
        assert type(df) == pd.DataFrame, 'must be a pd.DataFrame!'
        
        self.cols = list(df.columns)
        binarizers= []
        if self.group_sparsity:
            GroupMap = {}
            
        for col_idx in range(len(self.cols)):
            col = self.cols[col_idx]
            col_value = df[col]
            
            if col_value.isnull().sum() > 0:
                tmp_num_thresholds -= 1
                binarizers.append({                 # need to keep track of NaN for every column
                    'name': f"{col}_isNaN",
                    'col': col,
                    'threshold': np.nan,
                })                      
            
            col_value = col_value.dropna()      # drop NaN
            if len(col_value) == 0:
                raise ValueError(f"Column '{col}' has no non-NaN values!")
            vals = col_value.unique()         # count unique values
            vals_counts = col_value.value_counts()
            tmp_num_thresholds = self.max_num_thresholds_per_feature
            
            
            if len(vals) == 1:
                if self.group_sparsity:
                    GroupMap[col] = col_idx
                continue
            elif len(vals) < tmp_num_thresholds:
                tmp_num_thresholds = len(vals)
            p = np.ones(len(vals)) / len(vals)
            if self.sampling_weights == 'weighted':
                p = vals_counts / vals_counts.sum()
            elif self.sampling_weights != 'uniform':
                raise ValueError("sampling_weights must be either 'uniform' or 'weighted'")
            sampled_indices = self.rng.choice(len(vals), tmp_num_thresholds, replace=False, p=p)
            perctile = vals[sampled_indices]
            perctile.sort()
            
            if self.whether_interval:
                # do it in intervals
                for i in range(0, len(perctile)):   # NOTE: changed due to inconsistency issue with number of bins
                    if i == 0:
                        name = f"{col}<={perctile[i]}"
                        threshold = perctile[i]
                    else:
                        name = f"{perctile[i-1]}<{col}<={perctile[i]}"
                        threshold = (perctile[i-1], perctile[i])
                    binarizers.append({
                        "name": name,
                        "col": col,
                        "threshold": threshold,           
                    })
                    if self.group_sparsity and col not in GroupMap.keys():
                        GroupMap[col] = col_idx
            else:
                # do it in <=
                for i in range(0, len(perctile)):   # NOTE: changed due to inconsistency issue with number of bins
                    binarizers.append({
                        "name": f"{col}<={perctile[i]}",
                        "col": col,
                        "threshold": perctile[i],           
                    })
                    if self.group_sparsity and col not in GroupMap.keys():
                        GroupMap[col] = col_idx
        
        if self.group_sparsity:
            assert len(GroupMap) == len(self.cols), "invalid GroupMap!"
            self.GroupMap = GroupMap
        self.binarizers = binarizers                # save binarizer and GroupMap for transform() function
    
    def transform(self, df: pd.DataFrame) -> tuple:
        """
        Transform data using percentiles found in fitting
        
        Parameters
        ----------
        df : pd.DataFrame
            data to transform
            
        Returns
        -------
        tuple
            transformed data, group sparsity index
        """
        assert hasattr(self, "binarizers"), 'IntervalBinarizer not fitted yet!'
        assert type(df) == pd.DataFrame, "must be a pd.DataFrame"
        assert list(self.cols) == list(df.columns), "data used for fitting and transforming must have same columns and same order!"
        
        n, result, GroupIdx = len(df), {}, []

        for bin in self.binarizers:
            feature = np.zeros(n, dtype=int)
            if self.whether_interval:
                if type(bin["threshold"]) != tuple:
                    if np.isnan(bin["threshold"]):
                        # if threshold is NaN, calculate NaN dummy
                        feature = nan_onehot_single_column(df[bin['col']])
                        result[bin['name']] = feature
                    else:
                        feature[df[bin["col"]] <= bin["threshold"]] = 1
                else:
                    # apply a < variable <= b, threshold determined by .fit()
                    feature[(bin["threshold"][0] < df[bin["col"]]) & (df[bin["col"]] <= bin["threshold"][1])] = 1
                
                if self.group_sparsity:
                    GroupIdx.append(self.GroupMap[bin["col"]])
                result[bin["name"]] = feature
            else:
                if np.isnan(bin['threshold']):
                    # if threshold is NaN, calculate NaN dummy
                    feature = nan_onehot_single_column(df[bin['col']])
                    result[bin['name']] = feature
                else:
                    feature[df[bin["col"]] <= bin["threshold"]] = 1
                    result[bin["name"]] = feature
                    
                if self.group_sparsity:
                    GroupIdx.append(self.GroupMap[bin["col"]])
        
        result = pd.DataFrame.from_dict(result)
        
        if self.group_sparsity:
            assert len(np.unique(GroupIdx)) == len(df.columns), "GroupIdx is wrong!"
            assert len(GroupIdx) == result.shape[1], "GroupIdx is wrong!"
            return result, np.asarray(GroupIdx)
        else:
            return result, None
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''fit and transform on same dataframe'''
        self.fit(df)
        return self.transform(df)