import numpy as np
import pandas as pd

def convert_continuous_df_to_binary_df(df, num_quantiles=100):
    """Convert a dataframe with continuous features to a dataframe with binary features by thresholding

    Parameters
    ----------
    df : pandas.DataFrame
        original dataframe where there are columns with continuous features
    num_quantiles : int, optional
        number of points we pick from quantile as thresholds if a column has too many unique values, by default 100

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

    for i in range(0, len(colnames)):
        uni = df[colnames[i]].unique()
        if len(uni) == 2:
            binarized_dict[colnames[i]] = np.asarray(df[colnames[i]], dtype=int)
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


    binarized_df = pd.DataFrame(binarized_dict)
    print("Finish converting continuous features to binary features......")
    return binarized_df