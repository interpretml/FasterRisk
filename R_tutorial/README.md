# Tutorial on Using FasterRisk Inside R

Author: Jiachang Liu

Date: Last Compiled on November 09, 2022

-   <a href="#installation" id="toc-installation">1 Installation</a>
    -   <a href="#install-the-reticulate-package"
        id="toc-install-the-reticulate-package">1.1 Install the reticulate
        Package</a>
    -   <a href="#create-a-python-virtual-environment"
        id="toc-create-a-python-virtual-environment">1.2 Create a Python Virtual
        Environment</a>
    -   <a href="#install-the-fasterrisk-package-from-pypi"
        id="toc-install-the-fasterrisk-package-from-pypi">1.3 Install the
        FasterRisk Package from PyPI</a>
-   <a href="#preparation-before-training"
    id="toc-preparation-before-training">2 Preparation before Training</a>
    -   <a href="#download-sample-data" id="toc-download-sample-data">2.1
        Download Sample Data</a>
    -   <a href="#read-sample-data" id="toc-read-sample-data">2.2 Read Sample
        Data</a>
-   <a href="#training-the-model" id="toc-training-the-model">3 Training the
    Model</a>
    -   <a href="#create-a-model-class" id="toc-create-a-model-class">3.1 Create
        A Model Class</a>
    -   <a href="#train-the-model" id="toc-train-the-model">3.2 Train the
        Model</a>
-   <a href="#get-risk-score-models" id="toc-get-risk-score-models">4 Get
    Risk Score Models</a>
    -   <a href="#get-solutions-from-the-trained-model-class"
        id="toc-get-solutions-from-the-trained-model-class">4.1 Get Solutions
        from the Trained Model Class</a>
    -   <a href="#access-the-first-risk-score-model"
        id="toc-access-the-first-risk-score-model">4.2 Access the First Risk
        Score Model</a>
    -   <a href="#use-the-first-risk-score-model-to-do-prediction"
        id="toc-use-the-first-risk-score-model-to-do-prediction">4.3 Use the
        First Risk Score Model to Do Prediction</a>
    -   <a href="#print-the-first-model-card"
        id="toc-print-the-first-model-card">4.4 Print the First Model Card</a>
    -   <a
        href="#print-top-10-model-cards-from-the-pool-and-their-performance-metrics"
        id="toc-print-top-10-model-cards-from-the-pool-and-their-performance-metrics">4.5
        Print Top 10 Model Cards from the Pool and Their Performance Metrics</a>

<!-- Rscript -e "rmarkdown::render('header.Rmd'); rmarkdown::render('README.Rmd')" -->

# 1 Installation

## 1.1 Install the reticulate Package

``` r
install.packages("reticulate", repos = "http://cran.us.r-project.org")
library(reticulate)
```

## 1.2 Create a Python Virtual Environment

``` r
version <- "3.9.12"
install_python(version)
virtualenv_create(envname="FasterRisk-environment", version = version)
use_virtualenv("FasterRisk-environment")
```

## 1.3 Install the FasterRisk Package from PyPI

``` r
py_install("fasterrisk", pip=TRUE, envname="FasterRisk-environment")
```

# 2 Preparation before Training

``` r
fasterrisk <- import("fasterrisk")
```

## 2.1 Download Sample Data

``` r
train_data_file_path <- "../tests/adult_train_data.csv"
test_data_file_path <- "../tests/adult_test_data.csv"

if (!file.exists(train_data_file_path)){
    fasterrisk$utils$download_file_from_google_drive('1nuWn0QVG8tk3AN4I4f3abWLcFEP3WPec', train_data_file_path)
}

if (!file.exists(train_data_file_path)){
    fasterrisk$utils$download_file_from_google_drive('1TyBO02LiGfHbatPWU4nzc8AndtIF-7WH', test_data_file_path)
}
```

## 2.2 Read Sample Data

``` r
np <- import("numpy", convert=FALSE)
train_df <- read.csv(train_data_file_path)
train_data <- data.matrix(train_df)
X_train <- np$array(train_data[, 2:ncol(train_data)])
y_train <- np$array(train_data[, 1], dtype=np$int)

test_df <- read.csv(test_data_file_path)
test_data <- data.matrix(test_df)
X_test <- np$array(test_data[, 2:ncol(test_data)])
y_test <- np$array(test_data[, 1], dtype=np$int)
```

# 3 Training the Model

## 3.1 Create A Model Class

``` r
sparsity <- as.integer(5)
parent_size <- as.integer(10)

RiskScoreOptimizer_m <- fasterrisk$fasterrisk$RiskScoreOptimizer(X = X_train, y = y_train, k = sparsity, parent_size = parent_size)
```

## 3.2 Train the Model

``` r
start_time <- Sys.time()
RiskScoreOptimizer_m$optimize()
sprintf("Optimization takes %f seconds.", Sys.time() - start_time)
```

    ## [1] "Optimization takes 12.420722 seconds."

# 4 Get Risk Score Models

## 4.1 Get Solutions from the Trained Model Class

``` r
solutions = RiskScoreOptimizer_m$get_models()
multipliers = solutions[1][[1]]
sparseDiversePool_beta0_integer = solutions[2][[1]]
sparseDiversePool_betas_integer = solutions[3][[1]]
sprintf("We generate %d risk score models from the sparse diverse pool", length(multipliers))
```

    ## [1] "We generate 50 risk score models from the sparse diverse pool"

## 4.2 Access the First Risk Score Model

``` r
model_index = 1 # first model
multiplier = multipliers[model_index]
intercept = sparseDiversePool_beta0_integer[model_index]
coefficients = np$array(sparseDiversePool_betas_integer[model_index, ])
```

## 4.3 Use the First Risk Score Model to Do Prediction

``` r
RiskScoreClassifier_m = fasterrisk$fasterrisk$RiskScoreClassifier(multiplier, intercept, coefficients)
```

``` r
y_test_pred = RiskScoreClassifier_m$predict(X_test)
print("y_test are predicted to be (first 10 values):")
```

    ## [1] "y_test are predicted to be (first 10 values):"

``` r
y_test_pred[1:10]
```

    ##  [1] -1 -1 -1 -1 -1 -1 -1  1 -1 -1

``` r
y_test_pred_prob = RiskScoreClassifier_m$predict_prob(X_test)
print("The risk probabilities of having y_test to be +1 are (first 10 values):")
```

    ## [1] "The risk probabilities of having y_test to be +1 are (first 10 values):"

``` r
y_test_pred_prob[1:10]
```

    ##  [1] 0.13308868 0.34872682 0.34872682 0.04216029 0.13308868 0.34872682
    ##  [7] 0.04216029 0.65127318 0.34872682 0.01246260

## 4.4 Print the First Model Card

``` r
X_featureNames = list(colnames(train_df)[-1])[[1]]

RiskScoreClassifier_m$reset_featureNames(X_featureNames)
tmp_str = py_capture_output(RiskScoreClassifier_m$print_model_card(), type = c("stdout", "stderr"))
cat(tmp_str)
```

    ## The Risk Score is:
    ## 1.            Age_22_to_29     -2 point(s) |   ...
    ## 2.               HSDiploma     -2 point(s) | + ...
    ## 3.                    NoHS     -4 point(s) | + ...
    ## 4.                 Married      4 point(s) | + ...
    ## 5.         AnyCapitalGains      3 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -8.0  |  -6.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |
    ## RISK  |   0.1% |   0.4% |   0.7% |   1.2% |   2.3% |   4.2% |   7.6% |
    ## SCORE |   0.0  |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |   7.0  |
    ## RISK  |  13.3% |  22.3% |  34.9% |  50.0% |  65.1% |  77.7% |  92.4% |

## 4.5 Print Top 10 Model Cards from the Pool and Their Performance Metrics

``` r
num_models = min(10, length(multipliers))

for (model_index in 1:num_models){
    multiplier = multipliers[model_index]
    intercept = sparseDiversePool_beta0_integer[model_index]
    coefficients = np$array(sparseDiversePool_betas_integer[model_index, ])

    RiskScoreClassifier_m = fasterrisk$fasterrisk$RiskScoreClassifier(multiplier, intercept, coefficients)
    RiskScoreClassifier_m$reset_featureNames(X_featureNames)
    # RiskScoreClassifier_m$print_model_card()
    tmp_str = py_capture_output(RiskScoreClassifier_m$print_model_card(), type = c("stdout", "stderr"))
    cat(tmp_str)

    train_loss = RiskScoreClassifier_m$compute_logisticLoss(X_train, y_train)
    train_results = RiskScoreClassifier_m$get_acc_and_auc(X_train, y_train)
    train_acc = train_results[1][[1]]
    train_auc = train_results[2][[1]]
    test_results = RiskScoreClassifier_m$get_acc_and_auc(X_test, y_test)
    test_acc = test_results[1][[1]]
    test_auc = test_results[2][[1]]

    tmp_str = sprintf("The logistic loss on the training set is %f", train_loss)
    print(tmp_str)
    tmp_str = sprintf("The training accuracy and AUC are %f and %f", train_acc*100, train_auc)
    print(tmp_str)
    tmp_str = sprintf("The test accuracy and AUC are are %f and %f", test_acc*100, test_auc)
    print(tmp_str)
    cat("\n")
}
```

    ## The Risk Score is:
    ## 1.            Age_22_to_29     -2 point(s) |   ...
    ## 2.               HSDiploma     -2 point(s) | + ...
    ## 3.                    NoHS     -4 point(s) | + ...
    ## 4.                 Married      4 point(s) | + ...
    ## 5.         AnyCapitalGains      3 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -8.0  |  -6.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |
    ## RISK  |   0.1% |   0.4% |   0.7% |   1.2% |   2.3% |   4.2% |   7.6% |
    ## SCORE |   0.0  |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |   7.0  |
    ## RISK  |  13.3% |  22.3% |  34.9% |  50.0% |  65.1% |  77.7% |  92.4% |
    ## 
    ## [1] "The logistic loss on the training set is 9798.652347"
    ## [1] "The training accuracy and AUC are 82.575147 and 0.861817"
    ## [1] "The test accuracy and AUC are are 81.787469 and 0.856367"
    ## 
    ## The Risk Score is:
    ## 1.               HSDiploma     -2 point(s) |   ...
    ## 2.                    NoHS     -4 point(s) | + ...
    ## 3.                 Married      4 point(s) | + ...
    ## 4.    WorkHrsPerWeek_lt_40     -2 point(s) | + ...
    ## 5.         AnyCapitalGains      3 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -8.0  |  -6.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |
    ## RISK  |   0.1% |   0.4% |   0.7% |   1.3% |   2.5% |   4.4% |   7.9% |
    ## SCORE |   0.0  |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |   7.0  |
    ## RISK  |  13.7% |  22.7% |  35.1% |  50.0% |  64.9% |  77.3% |  92.1% |
    ## 
    ## [1] "The logistic loss on the training set is 9859.615758"
    ## [1] "The training accuracy and AUC are 82.333295 and 0.859661"
    ## [1] "The test accuracy and AUC are are 81.848894 and 0.853752"
    ## 
    ## The Risk Score is:
    ## 1.               HSDiploma     -3 point(s) |   ...
    ## 2.                    NoHS     -5 point(s) | + ...
    ## 3.           JobManagerial      2 point(s) | + ...
    ## 4.                 Married      5 point(s) | + ...
    ## 5.         AnyCapitalGains      3 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -8.0  |  -6.0  |  -5.0  |  -3.0  |  -2.0  |  -1.0  |   0.0  |
    ## RISK  |   0.2% |   0.6% |   1.0% |   2.7% |   4.4% |   7.2% |  11.4% |
    ## SCORE |   2.0  |   3.0  |   4.0  |   5.0  |   7.0  |   8.0  |  10.0  |
    ## RISK  |  26.4% |  37.5% |  50.0% |  62.5% |  82.3% |  88.6% |  95.6% |
    ## 
    ## [1] "The logistic loss on the training set is 9883.324462"
    ## [1] "The training accuracy and AUC are 82.268033 and 0.859570"
    ## [1] "The test accuracy and AUC are are 81.511057 and 0.854237"
    ## 
    ## The Risk Score is:
    ## 1.               HSDiploma     -3 point(s) |   ...
    ## 2.                    NoHS     -5 point(s) | + ...
    ## 3.                 Married      5 point(s) | + ...
    ## 4.   WorkHrsPerWeek_geq_50      1 point(s) | + ...
    ## 5.         AnyCapitalGains      3 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -8.0  |  -7.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |   0.0  |
    ## RISK  |   0.2% |   0.3% |   0.9% |   1.5% |   2.5% |   4.1% |   6.8% |  10.9% |
    ## SCORE |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |   6.0  |   8.0  |   9.0  |
    ## RISK  |  17.2% |  25.9% |  37.2% |  50.0% |  62.8% |  74.1% |  89.1% |  93.2% |
    ## 
    ## [1] "The logistic loss on the training set is 9895.728068"
    ## [1] "The training accuracy and AUC are 82.179738 and 0.861227"
    ## [1] "The test accuracy and AUC are are 81.342138 and 0.856065"
    ## 
    ## The Risk Score is:
    ## 1.            Age_45_to_59      1 point(s) |   ...
    ## 2.               HSDiploma     -2 point(s) | + ...
    ## 3.                    NoHS     -5 point(s) | + ...
    ## 4.                 Married      4 point(s) | + ...
    ## 5.         AnyCapitalGains      3 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -7.0  |  -6.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |   0.0  |
    ## RISK  |   0.2% |   0.4% |   0.7% |   1.2% |   2.0% |   3.5% |   5.9% |   9.9% |
    ## SCORE |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |   6.0  |   7.0  |   8.0  |
    ## RISK  |  16.0% |  24.9% |  36.5% |  50.0% |  63.5% |  75.1% |  84.0% |  90.1% |
    ## 
    ## [1] "The logistic loss on the training set is 9914.759742"
    ## [1] "The training accuracy and AUC are 80.655687 and 0.862991"
    ## [1] "The test accuracy and AUC are are 80.052211 and 0.856397"
    ## 
    ## The Risk Score is:
    ## 1.               HSDiploma     -3 point(s) |   ...
    ## 2.                    NoHS     -5 point(s) | + ...
    ## 3.                 Married      5 point(s) | + ...
    ## 4.         AnyCapitalGains      3 point(s) | + ...
    ## 5.          AnyCapitalLoss      2 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -8.0  |  -6.0  |  -5.0  |  -3.0  |  -2.0  |  -1.0  |   0.0  |
    ## RISK  |   0.2% |   0.5% |   0.8% |   2.3% |   3.9% |   6.4% |  10.5% |
    ## SCORE |   2.0  |   3.0  |   4.0  |   5.0  |   7.0  |   8.0  |  10.0  |
    ## RISK  |  25.5% |  36.9% |  50.0% |  63.1% |  83.3% |  89.5% |  96.1% |
    ## 
    ## [1] "The logistic loss on the training set is 9923.881690"
    ## [1] "The training accuracy and AUC are 82.179738 and 0.857475"
    ## [1] "The test accuracy and AUC are are 81.342138 and 0.852014"
    ## 
    ## The Risk Score is:
    ## 1.               HSDiploma     -2 point(s) |   ...
    ## 2.             ProfVocOrAS     -1 point(s) | + ...
    ## 3.                    NoHS     -4 point(s) | + ...
    ## 4.                 Married      3 point(s) | + ...
    ## 5.         AnyCapitalGains      2 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -7.0  |  -6.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |
    ## RISK  |   0.1% |   0.1% |   0.3% |   0.8% |   1.7% |   3.7% |   8.0% |
    ## SCORE |   0.0  |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |
    ## RISK  |  16.4% |  30.7% |  50.0% |  69.3% |  83.6% |  92.0% |
    ## 
    ## [1] "The logistic loss on the training set is 9980.639484"
    ## [1] "The training accuracy and AUC are 82.172060 and 0.855975"
    ## [1] "The test accuracy and AUC are are 81.234644 and 0.849403"
    ## 
    ## The Risk Score is:
    ## 1.               HSDiploma     -2 point(s) |   ...
    ## 2.                    NoHS     -4 point(s) | + ...
    ## 3.                 Married      3 point(s) | + ...
    ## 4.            NeverMarried     -1 point(s) | + ...
    ## 5.         AnyCapitalGains      2 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -7.0  |  -6.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |
    ## RISK  |   0.1% |   0.3% |   0.6% |   1.2% |   2.4% |   4.9% |   9.8% |
    ## SCORE |   0.0  |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |
    ## RISK  |  18.6% |  32.3% |  50.0% |  67.7% |  81.4% |  90.2% |
    ## 
    ## [1] "The logistic loss on the training set is 9988.041002"
    ## [1] "The training accuracy and AUC are 82.179738 and 0.855476"
    ## [1] "The test accuracy and AUC are are 81.342138 and 0.849000"
    ## 
    ## The Risk Score is:
    ## 1.               HSDiploma     -2 point(s) |   ...
    ## 2.                    NoHS     -4 point(s) | + ...
    ## 3.                 Married      4 point(s) | + ...
    ## 4.     DivorcedOrSeparated      1 point(s) | + ...
    ## 5.         AnyCapitalGains      2 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -6.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |   0.0  |
    ## RISK  |   0.1% |   0.3% |   0.6% |   1.2% |   2.6% |   5.1% |  10.1% |
    ## SCORE |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |   6.0  |   7.0  |
    ## RISK  |  18.9% |  32.6% |  50.0% |  67.4% |  81.1% |  89.9% |  94.9% |
    ## 
    ## [1] "The logistic loss on the training set is 10000.803139"
    ## [1] "The training accuracy and AUC are 82.179738 and 0.854825"
    ## [1] "The test accuracy and AUC are are 81.342138 and 0.848115"
    ## 
    ## The Risk Score is:
    ## 1.               HSDiploma     -2 point(s) |   ...
    ## 2.                    NoHS     -4 point(s) | + ...
    ## 3.              JobService     -1 point(s) | + ...
    ## 4.                 Married      4 point(s) | + ...
    ## 5.         AnyCapitalGains      3 point(s) | + ...
    ##                                      SCORE | =    
    ## SCORE |  -7.0  |  -6.0  |  -5.0  |  -4.0  |  -3.0  |  -2.0  |  -1.0  |   0.0  |
    ## RISK  |   0.2% |   0.4% |   0.7% |   1.3% |   2.4% |   4.3% |   7.7% |  13.4% |
    ## SCORE |   1.0  |   2.0  |   3.0  |   4.0  |   5.0  |   6.0  |   7.0  |
    ## RISK  |  22.4% |  35.0% |  50.0% |  65.0% |  77.6% |  86.6% |  92.3% |
    ## 
    ## [1] "The logistic loss on the training set is 10000.838407"
    ## [1] "The training accuracy and AUC are 82.175899 and 0.857572"
    ## [1] "The test accuracy and AUC are are 81.464988 and 0.851669"
