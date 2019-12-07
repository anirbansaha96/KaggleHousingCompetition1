import pandas as pd
from sklearn.model_selection import train_test_split
X=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#We Drop Rows with missing SalePrice in training set
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing1 = [col for col in X.columns if X[col].isnull().any()]
cols_with_missing2 = [col for col in test.columns if test[col].isnull().any()]
cols_with_missing=cols_with_missing1+cols_with_missing2
X.drop(cols_with_missing, axis=1, inplace=True)
test.drop(cols_with_missing, axis=1, inplace=True)

#Train Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)


#Making a list of columns with datatype object
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)


#Starting with One Hot Encoding
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(OH_X_train, y_train)
preds = model.predict(OH_X_valid)
print(mean_absolute_error(y_valid, preds))


#Generating Test Results
OH_cols_test = pd.DataFrame(OH_encoder.transform(test[low_cardinality_cols]))
OH_cols_test.index = test.index
num_X_test = test.drop(object_cols, axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
preds2 = model.predict(OH_X_test)
output = pd.DataFrame({'Id': test.index,
                       'SalePrice': preds2})
output.to_csv('submission.csv', index=False)