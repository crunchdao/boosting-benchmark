@info "This script tests the lightgbm installation and set-up"
## https://www.juliapackages.com/p/lightgbm

using LightGBM
using DelimitedFiles

LIGHTGBM_SOURCE =  relpath("..//LightGBM-3.2.0")  ## abspath(pwd()*"/LightGBM-3.2.0")

# Load LightGBM's binary classification example.
binary_test = readdlm(joinpath(LIGHTGBM_SOURCE, "examples", "binary_classification", "binary.test"), '\t')
binary_train = readdlm(joinpath(LIGHTGBM_SOURCE, "examples", "binary_classification", "binary.train"), '\t')
X_train = binary_train[:, 2:end]
y_train = binary_train[:, 1]
X_test = binary_test[:, 2:end]
y_test = binary_test[:, 1]

# Create an estimator with the desired parameters—leave other parameters at the default values.
estimator = LGBMClassification(
    objective = "binary",
    num_iterations = 100,
    learning_rate = .1,
    early_stopping_round = 5,
    feature_fraction = .8,
    bagging_fraction = .9,
    bagging_freq = 1,
    num_leaves = 1000,
    num_class = 1,
    metric = ["auc", "binary_logloss"]
)

# Fit the estimator on the training data and return its scores for the test data.
fit!(estimator, X_train, y_train, (X_test, y_test))

# Predict arbitrary data with the estimator.
predict(estimator, X_train)

