# Major-Assignment-1

Phase 1- Used Machine Appendix to use other features and analysed the features.

Phase 2- I analysed the data, the datatypes and used a threshold of 80% to drop a column. Used simple imputer for missing values, initally used KNN but the computation was very slow.

Imputed both train and test categories with missing value. Used ordinal encoder and one hot encoding for categories.

Dropped columns with more than 60% data missing and aligned test and train columns. If a column is present in train and not present in test, I imputed with median of train data.

Phase 3- Applied XGBoost along with optuna (read the documentation ) for tuning hyperparamters and Random forest and compared their result with R2 and RMSE. Applied ridge regression with normalisation but got inferior results.

Phase 4- Used test file for predictions.

# Major Assignment-2
