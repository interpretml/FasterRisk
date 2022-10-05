# fasterrisk

Create sparse and accurate risk scoring systems!

## Installation

```bash
$ pip install fasterrisk
```

## Package Development ToDo List (ongoing)
- [ ] Reupload data to Google Drive with .csv files and colnames included
- [ ] Implement print_model_card() function in the fasterrisk module.
- [ ] Add the usage of print_model_card() in example jupyter notebook.
- [ ] Upload to TestPyPI to test the package
- [ ] Add citation bib file at the bottom once the article is available on Google Scholar
- [ ] Add underscore in front of internal functions in the modules?
- [ ] Host documentation online with Read the Docs
- [ ] Add Read the Docs link to GitHub repo README once it is available.

## Usage

Two classes:
- RiskScoreOptimizer
```python
# initialize a risk score optimizer
m = RiskScoreOptimizer(X_train=X_train, y_train=y_train, k=k)

# perform optimization
m.optimize()

# get all top m solutions from the final diverse pool
multipliers, intercepts, coefficients = m.get_models(model_index=None) # get m solutions from the diverse pool; Specifically, multipliers.shape=(m, ), intercepts.shape=(m, ), coefficients.shape=(m, p)

# get the first solution from the final diverse pool by passing an optional model_index; models are ranked in order of increasing logistic loss
multiplier, intercept, coefficient = m.get_models(model_index=0) # get the first solutions from the diverse pool; Specifically, multiplier.shape=(1, ), intercept.shape=(1, ), coefficients.shape=(p, )

# print all model cards from the final diverse pool
m.print_model_card(model_index=None) 

# print the first model card from the final diverse pool
m.print_model_card(model_index=0) 
```

- RiskScoreClassifier
```python
# create a classifier
clf = RiskScoreClassifier(multiplier=multiplier, intercept=intercept, coefficients=coefficients)

# get the predicted label
y_pred = clf.predict(X = X_test)

# get the probability of predicting y[i] with label +1
y_pred_prob = clf.predict_prob(X = X_test)

# compute the logistic loss
logisticLoss = clf.compute_logisticLoss(X = X_train, y = y_train)
```


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`fasterrisk` was created by Jiachang Liu. It is licensed under the terms of the BSD 3-Clause license.

## Credits

`fasterrisk` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
