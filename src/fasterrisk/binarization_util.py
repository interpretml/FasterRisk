from typing import *

import numpy as np
import pandas as pd
from sklearn.preprocessing._encoders import _BaseEncoder


def convert_continuous_df_to_binary_df(df, num_quantiles=100, get_featureIndex_to_groupIndex=False):
    """Convert a dataframe with continuous features to a dataframe with binary features by thresholding

    Parameters
    ----------
    df : pandas.DataFrame
        original dataframe where there are columns with continuous features
    num_quantiles : int, optional
        number of points we pick from quantile as thresholds if a column has too many unique values, by default 100
    get_featureIndex_to_groupIndex : bool, optional
        whether to return a numpy array that maps feature index to group index, by default False

    Returns
    -------
    binarized_df : pandas.DataFrame
        a new dataframe where each column only has 0/1 as the feature
    """

    colnames = df.columns
    n = len(df)
    print("Converting continuous features to binary features in the dataframe......")
    print("If a feature has more than 100 unqiue values, we pick the threasholds by selecting 100 quantile points. You can change the number of thresholds by passing another specified number: convert_continuous_df_to_binary_df(df, num_quantiles=50).")

    percentile_ticks = range(1, num_quantiles+1)

    binarized_dict = {}

    featureIndex_to_groupIndex = []
    for i in range(0, len(colnames)):
        uni = df[colnames[i]].unique()
        if len(uni) == 2:
            binarized_dict[colnames[i]] = np.asarray(df[colnames[i]], dtype=int)
            featureIndex_to_groupIndex.append(i)
            continue

        uni.sort()
        if len(uni) >= 100:
            uni = np.percentile(uni, percentile_ticks)
        for j in range(len(uni)-1):
            tmp_feature = np.ones(n, dtype=int)
            tmp_name = colnames[i] + "<=" + str(uni[j])

            zero_indices = df[colnames[i]] > uni[j]
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
    def __init__(self, interval_width: int, whether_interval: bool=False, group_sparsity: bool=False) -> None:
        assert type(interval_width) == int, "'interval_width' must be integer!"
        assert 100 % interval_width == 0, "'interval_width' must divide 100!"
        self.interval_width = interval_width
        self.whether_interval = whether_interval
        self.group_sparsity = group_sparsity
    
    def fit(self, df: pd.DataFrame) -> None:
        '''fit IntervalBinarizer'''
        assert type(df) == pd.DataFrame, 'must be a pd.DataFrame!'
        
        self.cols = list(df.columns)
        tiles = range(self.interval_width, 100+self.interval_width, self.interval_width)
        binarizers= []
        if self.group_sparsity:
            GroupMap = {}
            
        for col_idx in range(len(self.cols)):
            col = self.cols[col_idx]
            col_value = df[col]
            
            binarizers.append({                 # need to keep track of NaN for every column
                'name': f"{col}_isNaN",
                'col': col,
                'threshold': np.nan,
            })                      
            
            col_value = col_value.dropna()      # drop NaN
            vals = col_value.unique()         # count unique values
            
            if len(vals) == 1: 
                if self.group_sparsity:
                    GroupMap[col] = col_idx
                continue
            elif len(vals) > (100/self.interval_width):     # if more than number of bins, do percentiles
                perctile = np.percentile(vals, tiles, method="closest_observation")
            else:   # else just use the unique values in sorted order
                perctile = np.sort(vals)
            
            if self.whether_interval:       # do it in intervals
                for i in range(0, len(perctile)):
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
            else:       # do it in <=
                for i in range(0, len(perctile)):
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
                    feature[df[bin["col"]] <= bin["threshold"]] = 1
                else:   # apply a < variable <= b, threshold determined by .fit()
                    feature[(bin["threshold"][0] < df[bin["col"]]) & (df[bin["col"]] <= bin["threshold"][1])] = 1
                
                if self.group_sparsity:
                    GroupIdx.append(self.GroupMap[bin["col"]])
                result[bin["name"]] = feature
            else:
                if np.isnan(bin['threshold']):                      # if threshold is NaN, calculate NaN dummy
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