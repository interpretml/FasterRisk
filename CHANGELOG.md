# Changelog

<!--next-version-placeholder-->
## v0.1.8 (07/23/2024)

- Add the functionality of obtaining groupIndex from binary feature names.
- Add the functionality of exporting the models as dictionaries. This allows users to save the models as JSON files and load them later.
- Fix the bug where the number of quantile points are not selected correctly.
- Add the functionality of printing more aesthetic risk score tables. Thanks to Muhang Tian for adding this feature!
- Add the functionality of training risk scoring systems with customized lower and upper bounds on the coefficients. This allows users to create monotonically increasing or decreasing risk scores. Thanks to Muhang Tian for adding this feature!
- Fix the bug where during rounding, two continuous solutions are rounded to the same solution (whose support is a strict subset of that of the continuous solutions). Thanks to Matt Oddo for finding this bug!

## v0.1.7 (04/03/2023)

- Add the functionality of training group sparsity-constrained risk scoring system

## v0.1.6 (03/12/2023)

- Fix example jupyter notebook file downloading issue
- Allow users to print out quantile points (default=20) for total risk scores when printing out the risk score tables.

## v0.1.5 (12/18/2022)

- Add a tutorial for R users.
- Fix the bug when parent_size is bigger than total number of children added to the forbidden support.
- Add binarization as part of the tutorial. Thanks Lisa Fretschel for reaching out and requesting this feature!

## v0.1.2 (10/23/2022)

- Fix sparseDiversePool so that it can handle X_train with a feature column with entries all equal to 1. Thanks Wes Jackson for finding this bug!
- Print top 10 models in example.ipynb.

## v0.1.1 (10/12/2022)

- First release of `fasterrisk`!