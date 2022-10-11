# fasterrisk

Create sparse and accurate risk scoring systems!

## Installation

```bash
conda create -n FasterRisk python=3.9 # create a virtual environment
conda activate FasterRisk # activate the virtual environment
python -m pip install fasterrisk # pip install the fasterrisk package
```

## Package Development ToDo List (ongoing)
- [x] Fix the intercept boolean parameter in logRegModel module
- [x] Reupload data to Google Drive with .csv files and colnames included
- [x] Implement print_model_card() function in the fasterrisk module.
- [x] Add the usage of print_model_card() in example jupyter notebook.
- [x] Fix un-transpose the solutions returned by sparseDiversePool 
- [x] Revise Usage section writeup in the README
- [x] Add conda environment creation terminal command 
- [x] To get models, implement sorting according to logistic loss. Has an internal flag to avoid sorting every time when calling this function
- [ ] Upload to TestPyPI to test the package
- [ ] Add citation bib file at the bottom once the article is available on Google Scholar
- [ ] Host documentation online with Read the Docs
- [ ] Add Read the Docs link to GitHub repo README once it is available.

## Usage
Please see the [example.ipynb](./docs/example.ipynb) jupyter notebook for a detailed tutorial on how to use FasterRisk in a python environment.

Two classes:
- RiskScoreOptimizer
```python
sparsity = 5 # produce a risk score model with 5 nonzero coefficients 

# import data
X_train, y_train = ...

# initialize a risk score optimizer
m = RiskScoreOptimizer(X = X_train, y = y_train, k = sparsity)

# perform optimization
m.optimize()

# get all top m solutions from the final diverse pool
arr_multiplier, arr_intercept, arr_coefficients = m.get_models() # get m solutions from the diverse pool; Specifically, arr_multiplier.shape=(m, ), arr_intercept.shape=(m, ), arr_coefficients.shape=(m, p)

# get the first solution from the final diverse pool by passing an optional model_index; models are ranked in order of increasing logistic loss
multiplier, intercept, coefficients = m.get_models(model_index = 0) # get the first solutions from the diverse pool; Specifically, multiplier.shape=(1, ), intercept.shape=(1, ), coefficients.shape=(p, )

```

- RiskScoreClassifier
```python
# import data
X_test, y_test, X_featureNames = ... # X_featureNames is a list of strings, each of which is the feature name

# create a classifier
clf = RiskScoreClassifier(multiplier = multiplier, intercept = intercept, coefficients = coefficients, featureNames = featureNames)

# get the predicted label
y_pred = clf.predict(X = X_test)

# get the probability of predicting y[i] with label +1
y_pred_prob = clf.predict_prob(X = X_test)

# compute the logistic loss
logisticLoss_train = clf.compute_logisticLoss(X = X_train, y = y_train)

# get accuracy and area under the ROC curve (AUC)
acc_test, auc_test = clf.get_acc_and_auc(X = X_test, y = y_test) 

# print the risk score model card
m.print_model_card() 
```


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`fasterrisk` was created by Jiachang Liu. It is licensed under the terms of the BSD 3-Clause license.

## Credits

`fasterrisk` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
