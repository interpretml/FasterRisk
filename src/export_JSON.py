###############################################################################

import os
import json
import numpy as np
import pandas as pd
from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

CSV_files = [
    # "data_input_CSV/adult_data",
    # "data_input_CSV/bank_data",
    # "data_input_CSV/fico_data",
    # "data_input_CSV/mammo_data",
    "data_input_CSV/shroom_data",
]
data_source = CSV_files[0]

###############################################################################

sparsity = 5  # default is 5
select_top_m = 200  # default is 50
gap_tolerance = 0.1  # default is 0.05 (5%)
parent_size = 10  # default is 10
maxAttempts = 50  # default is 50

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

train_df = pd.read_csv(f"{data_source}.csv")
train_data = np.asarray(train_df)
X_train = train_data[:, 1:]
y_train = train_data[:, 0]
y_train = np.where(y_train != 1, -1, y_train)

RiskScoreOptimizer_m = RiskScoreOptimizer(
    X=X_train,
    y=y_train,
    k=sparsity,
    parent_size=parent_size,
    select_top_m=select_top_m,
    gap_tolerance=gap_tolerance,
    maxAttempts=maxAttempts,
)

RiskScoreOptimizer_m.optimize()
(
    multipliers,
    sparseDiversePool_beta0_integer,
    sparseDiversePool_betas_integer,
) = RiskScoreOptimizer_m.get_models()

###############################################################################

model_index = 0
multiplier = multipliers[model_index]
intercept = sparseDiversePool_beta0_integer[model_index]
coefficients = sparseDiversePool_betas_integer[model_index]

RiskScoreClassifier_m = RiskScoreClassifier(
    multiplier,
    intercept,
    coefficients,
    X_train=X_train,
)

X_featureNames = list(train_df.columns[1:])

RiskScoreClassifier_m.reset_featureNames(X_featureNames)

print("\nNumber of Cards: ", len(multipliers), "\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

FasterRisk_cards = RiskScoreOptimizer_m.get_models_in_dict(
    X_featureNames,
    featureIndex_to_groupIndex=np.arange(len(multipliers), dtype=int),
)

###############################################################################

filename = data_source.split("/")[-1]

with open(
    f"data_output_JSON/{filename}_{sparsity}-{select_top_m}-{gap_tolerance}-{parent_size}-{maxAttempts}.json",
    "w",
) as f:
    json.dump(FasterRisk_cards, f)

###############################################################################
