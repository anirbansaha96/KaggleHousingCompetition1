################################################################################################
import pandas as pd
import os
os.chdir('C:\\Users\\anirb\\Downloads\\house-prices-advanced-regression-techniques')
train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)
################################################################################################

#############################               Finding Null Values                  ######################################
null_columns_train=train.columns[train.isnull().any()]
number_null_train=train[null_columns_train].isnull().sum()
null_columns_test=test.columns[test.isnull().any()]
number_null_test=test[null_columns_test].isnull().sum()

null_columns_train_drop=null_columns_train[number_null_train>0.7*len(train.index)]
null_columns_train_drop=set(null_columns_train_drop)
null_columns_test_drop=null_columns_test[number_null_test>0.7*len(test.index)]
null_columns_test_drop=set(null_columns_test_drop)
drop_cols=null_columns_test_drop.union(null_columns_train_drop)
drop_cols=list(drop_cols)

train=train.drop(drop_cols, axis=1);
test=test.drop(drop_cols, axis=1);

null_columns_train=train.columns[train.isnull().any()]
number_null_train=train[null_columns_train].isnull().sum()
null_columns_test=test.columns[test.isnull().any()]
number_null_test=test[null_columns_test].isnull().sum()
################################################################################################


#########################        Filling Null Values with Modal Values        ##################################################
train[null_columns_train] = train[null_columns_train].fillna(train[null_columns_train].mode().iloc[0])
test[null_columns_test] = test[null_columns_test].fillna(test[null_columns_test].mode().iloc[0])
################################################################################################

X=train[['LotFrontage','OverallQual','OverallCond','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','TotRmsAbvGrd','GarageArea','PoolArea']]
X_out=test[['LotFrontage','OverallQual','OverallCond','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','TotRmsAbvGrd','GarageArea','PoolArea']]

##### FEATURE SCALING #####
from sklearn.preprocessing import StandardScaler
StScaler=StandardScaler()
X=StScaler.fit_transform(X)
X_out=StScaler.transform(X_out)
###########################
y=train['SalePrice']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#sc_y = StandardScaler()
#y_train=pd.DataFrame(y_train)
#y_train = sc_y.fit_transform(y_train.reshape(-1,1))
################################################################################################
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

output = regressor.predict(X_out)
output1=pd.DataFrame(data=output,
                    index=test.index,
                    columns={'SalePrice'})
output1.to_csv('file_submit1.csv') 

