#Importing basic libraries
import numpy as np 
import pandas as pd 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

import os
print(os.listdir("../input"))
input_data = pd.read_csv("../input/train.csv")

#Dataframe for NAN value handling
null_values=pd.DataFrame({'Column': {0: 'MSZoning',
  1: 'LotFrontage',
  2: 'Alley',
  3: 'Utilities',
  4: 'Exterior1st',
  5: 'Exterior2nd',
  6: 'MasVnrType',
  7: 'MasVnrArea',
  8: 'BsmtQual',
  9: 'BsmtCond',
  10: 'BsmtExposure',
  11: 'BsmtFinType1',
  12: 'BsmtFinSF1',
  13: 'BsmtFinType2',
  14: 'BsmtFinSF2',
  15: 'BsmtUnfSF',
  16: 'TotalBsmtSF',
  17: 'BsmtFullBath',
  18: 'BsmtHalfBath',
  19: 'KitchenQual',
  20: 'Electrical',
  21: 'Functional',
  22: 'FireplaceQu',
  23: 'GarageType',
  24: 'GarageYrBlt',
  25: 'GarageFinish',
  26: 'GarageCars',
  27: 'GarageArea',
  28: 'GarageQual',
  29: 'GarageCond',
  30: 'PoolQC',
  31: 'Fence',
  32: 'MiscFeature',
  33: 'SaleType'},
 'data type': {0: 'Nominal',
  1: 'ratio',
  2: 'Nominal',
  3: 'Ordinal',
  4: 'Nominal',
  5: 'exclude',
  6: 'Nominal',
  7: 'ratio',
  8: 'Ordinal',
  9: 'Ordinal',
  10: 'Ordinal',
  11: 'Ordinal',
  12: 'ratio',
  13: 'Ordinal',
  14: 'ratio',
  15: 'ratio',
  16: 'ratio',
  17: 'ratio',
  18: 'ratio',
  19: 'Ordinal',
  20: 'Nominal',
  21: 'Nominal',
  22: 'Ordinal',
  23: 'Nominal',
  24: 'exclude',
  25: 'Ordinal',
  26: 'exclude',
  27: 'ratio',
  28: 'Ordinal',
  29: 'Ordinal',
  30: 'Ordinal',
  31: 'Ordinal',
  32: 'Nominal',
  33: 'Nominal'},
 'dtype': {0: 'object',
  1: 'float64',
  2: 'object',
  3: 'object',
  4: 'object',
  5: 'object',
  6: 'object',
  7: 'float64',
  8: 'object',
  9: 'object',
  10: 'object',
  11: 'object',
  12: 'float64',
  13: 'object',
  14: 'float64',
  15: 'float64',
  16: 'float64',
  17: 'float64',
  18: 'float64',
  19: 'object',
  20: 'object',
  21: 'object',
  22: 'object',
  23: 'object',
  24: 'float64',
  25: 'object',
  26: 'float64',
  27: 'float64',
  28: 'object',
  29: 'object',
  30: 'object',
  31: 'object',
  32: 'object',
  33: 'object'},
 'fill value': {0: 'RL',
  1: 0,
  2: 'None',
  3: 'AllPub',
  4: 'VinylSd',
  5: 'exclude',
  6: 'None',
  7: 0,
  8: 'None',
  9: 'None',
  10: 'None',
  11: 'None',
  12: 0,
  13: 'None',
  14: 0,
  15: 0,
  16: 0,
  17: 0,
  18: 0,
  19: 'TA',
  20: 'Mix',
  21: 'Typ',
  22: 'None',
  23: 'None',
  24: 'exclude',
  25: 'None',
  26: 'exclude',
  27: 0,
  28: 'None',
  29: 'None',
  30: 'None',
  31: 'None',
  32: 'None',
  33: 'WD'}})


#Dataframe for categorical handling
categorical_info=pd.DataFrame({'Column': {0: 'Street',
  1: 'Alley',
  2: 'LotShape',
  3: 'LandContour',
  4: 'Utilities',
  5: 'LotConfig',
  6: 'LandSlope',
  7: 'Neighborhood',
  8: 'Condition1',
  9: 'Condition2',
  10: 'BldgType',
  11: 'HouseStyle',
  12: 'RoofStyle',
  13: 'RoofMatl',
  14: 'Exterior1st',
  15: 'Exterior2nd',
  16: 'MasVnrType',
  17: 'ExterQual',
  18: 'ExterCond',
  19: 'Foundation',
  20: 'BsmtQual',
  21: 'BsmtCond',
  22: 'BsmtExposure',
  23: 'BsmtFinType1',
  24: 'BsmtFinType2',
  25: 'Heating',
  26: 'HeatingQC',
  27: 'CentralAir',
  28: 'Electrical',
  29: 'KitchenQual',
  30: 'Functional',
  31: 'FireplaceQu',
  32: 'GarageType',
  33: 'GarageFinish',
  34: 'GarageQual',
  35: 'GarageCond',
  36: 'PavedDrive',
  37: 'PoolQC',
  38: 'Fence',
  39: 'MiscFeature',
  40: 'SaleType',
  41: 'SaleCondition',
                                         42: 'MSZoning'},
 'Nominal_ordinal': {0: 'N',
  1: 'N',
  2: 'N',
  3: 'N',
  4: 'O',
  5: 'N',
  6: 'N',
  7: 'N',
  8: 'N',
  9: 'N',
  10: 'N',
  11: 'N',
  12: 'N',
  13: 'N',
  14: 'N',
  15: 'N',
  16: 'N',
  17: 'O',
  18: 'O',
  19: 'N',
  20: 'O',
  21: 'O',
  22: 'O',
  23: 'O',
  24: 'O',
  25: 'N',
  26: 'O',
  27: 'N',
  28: 'N',
  29: 'O',
  30: 'N',
  31: 'O',
  32: 'N',
  33: 'O',
  34: 'O',
  35: 'O',
  36: 'N',
  37: 'O',
  38: 'O',
  39: 'N',
  40: 'N',
  41: 'N',
                    42:'N'},
 'encoding': {0: '-',
  1: '-',
  2: '-',
  3: '-',
  4: 'rank hot',
  5: '-',
  6: '-',
  7: '-',
  8: '-',
  9: '-',
  10: '-',
  11: '-',
  12: '-',
  13: '-',
  14: '-',
  15: '-',
  16: '-',
  17: 'decimal',
  18: 'decimal',
  19: '-',
  20: 'decimal',
  21: 'decimal',
  22: 'decimal',
  23: 'decimal',
  24: 'decimal',
  25: '-',
  26: 'decimal',
  27: '-',
  28: '-',
  29: 'decimal',
  30: '-',
  31: 'decimal',
  32: '-',
  33: 'decimal',
  34: 'decimal',
  35: 'decimal',
  36: '-',
  37: 'decimal',
  38: 'decimal',
  39: '-',
  40: '-',
  41: '-',
             42: '-'}})

ordinal_categories_dict={'BsmtCond': 'None_Ex_dict',
 'BsmtExposure': 'BsmtExposure_dict',
 'BsmtFinType1': 'BsmtFinType_dict',
 'BsmtFinType2': 'BsmtFinType_dict',
 'BsmtQual': 'BsmtQual_dict',
 'ExterCond': 'None_Ex_dict',
 'ExterQual': 'None_Ex_dict',
 'Fence': 'Fence_dict',
 'FireplaceQu': 'None_Ex_dict',
 'GarageCond': 'None_Ex_dict',
 'GarageFinish': 'GarageFinish_dict',
 'GarageQual': 'None_Ex_dict',
 'HeatingQC': 'None_Ex_dict',
 'KitchenQual': 'None_Ex_dict',
 'PoolQC': 'None_Ex_dict', 'Utilities' :'Utilities_dict'
 }

#Drop ID columns
input_data_without_Id=input_data.drop('Id',axis=1)

#Extract a series for NAN value handling
null_value_series=pd.Series(null_values['fill value'].values,index=null_values['Column'].values)
excluded_column_list=list(null_values['Column'][null_values['fill value']=='exclude'])
null_value_series.drop(axis=0,index=list(null_values['Column'][null_values['fill value']=='exclude']),inplace=True)


# **Imputation**

from sklearn.base import TransformerMixin

class Null_filler(TransformerMixin):
    def __init__(self):
        return None

    def fit(self,df,y=None,**fit_params):
        return self
    
    def transform(self,df,**transform_params):
        df_copy=df.copy()
        df_copy.fillna(value=null_value_series.to_dict(),inplace=True)
        df_copy.drop(excluded_column_list,axis=1,inplace=True)
        
        
        if df_copy.isna().any().any():
            #df_copy.dropna(axis=0,inplace=True)
            raise ValueError('Null Values exist in the data.')
            my_imputer = Imputer()
            df_copy = my_imputer.fit_transform(df_copy)
            
        return df_copy


# import seaborn as sns
# from matplotlib import pyplot as plt

# test_data=pd.read_csv('../input/test.csv')
# 

# sns.boxplot(input_data_without_Id['YrSold'],input_data_without_Id['SaleType'])

# sns.set(rc={'figure.figsize':(24,24)})
# ax=sns.heatmap(temp.corr(),annot=True,fmt=".2f",center=0)
# 

# **Encoder**



categorical_columns=categorical_info['Column']
nominal_categories=list(categorical_columns[categorical_info['Nominal_ordinal']=='N'].values)
ordinal_categories=list(categorical_columns[categorical_info['Nominal_ordinal']=='O'].values)

#Ensure that the columns to be dropped are not included in the columns to be encoded
for item in excluded_column_list:
    try:
        nominal_categories.remove(item)
        ordinal_categories.remove(item)
    except ValueError:
        pass
    

#One hot encoding using pandas_get_dummies for nominal columns
class dummy_encoder(TransformerMixin):
    def __init__(self):
        self.aligner=None
        return None

    def fit(self,df,**fit_params):
        df_copy=df.copy()
        #print(df_copy.columns)
        df_copy[df_copy[nominal_categories]=='None']=np.nan
        
        dummies=pd.get_dummies(df_copy[nominal_categories])
        #print(dummies)
        df_copy.drop(nominal_categories,axis=1,inplace=True)
        
        self.aligner=pd.concat([df_copy,dummies],axis=1).sample(10)
        return self
    
    def transform(self,df,**transform_params):
        if type(self.aligner) is type(None):
            raise ValueError("Must fit before transform")
        df_copy=df.copy()
        df_copy[df_copy[nominal_categories]=='None']=np.nan
        dummies=pd.get_dummies(df_copy[nominal_categories])
        #print(dummies)
        df_copy.drop(nominal_categories,axis=1,inplace=True)
        
        concantenated_df=pd.concat([df_copy,dummies],axis=1)
        
        (_,aligned_df)=self.aligner.align(concantenated_df,join='left',axis=1,fill_value=0)
        return aligned_df



import copy
from math import log,ceil,floor

#Function to convert an integer to a binary string separated by commas
def int_to_bin(number,total_bits):
    temp_list=[str(i) for i in bin(number).split('b')[-1]]
    final_encoded=['0']*(total_bits-len(temp_list))+temp_list
    
    return ','.join(final_encoded)

class binary_encoder(TransformerMixin):
    """Custom encoder for having userdefined label for each category"""
    def __init__(self):
        self.lut=dict()
        #self._dim=None
        self.column=None
        self.return_type='decimal'
        return None

    def fit(self,df,y=None,**fit_params):
        self.column=fit_params['column']
        #self.return_type=fit_params['return_type']
        #self._dim=df[self.column].shape[0]
        extracted_unique_values=df[self.column].unique()
        if y:
            for value in extracted_unique_values:
                if value not in y:
                    raise KeyError('Dataframe'+self.column+'has more unique values than specified in the look-up table.')
            self.lut=y.copy()
        else:
            self.lut=dict((j,i) for i,j in enumerate(extracted_unique_values))
        
        return self
    
    
    
    def transform(self,df,**transform_params):
        
        if type(self.column) is type(None):
            raise ValueError('Must Fit before transform.')
        df_copy=df.copy()
        
        new_series=df_copy[self.column].copy()
        

        if new_series.isna().any():
            raise ValueError('There are NaN in data. Apply imputer before tranform.')
        for key in self.lut:
            new_series[new_series==key]=self.lut[key]

            
        if self.return_type=='decimal':
            df_copy[self.column]=new_series
            concantenated_df=df_copy
            
        elif self.return_type=='rank hot':
            num=max([j for i,j in self.lut.items()])
            total_bits=floor(log(num,2))+1

            new_series_b_encoded=new_series.apply(lambda x:int_to_bin(x,total_bits))
            new_column_names=[new_series.name+'_'+str(i) for i in range(total_bits)]

            new_df_b_encoded=new_series_b_encoded.str.split(pat=',',expand=True).apply(pd.to_numeric)
            new_df_b_encoded.set_axis(new_column_names,axis=1,inplace=True)


            df_copy.drop(self.column,axis=1,inplace=True)
            concantenated_df=pd.concat([df_copy,new_df_b_encoded],axis=1,sort=False)
        else:
            raise ValueError('Invalid type specified for categorical encoding')
        
        return concantenated_df


#Encoder for ordinal categories

class ordinal_encoder(TransformerMixin):
    def __init__(self):
        self.dict_of_dicts={'BsmtExposure_dict':{'Av': 3, 'Gd': 4, 'Mn': 2, 'No': 1, 'None': 0},
                           'BsmtFinType_dict':{'ALQ': 5, 'BLQ': 4, 'GLQ': 6, 'LwQ': 2, 'None': 0, 'Rec': 3, 'Unf': 1},
                           'Fence_dict':{'GdPrv': 4, 'GdWo': 2, 'MnPrv': 3, 'MnWw': 1, 'None': 0},
                           'GarageFinish_dict':{'Fin': 3, 'None': 0, 'RFn': 2, 'Unf': 1},
                           'Utilities_dict':{'AllPub': 3, 'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2},
                           'None_Ex_dict':{'None': 0,'Po': 1, 'Fa': 2, 'TA': 3,'Gd': 4,  'Ex': 5 },
                           'BsmtQual_dict':{'None': 0,'Po': 35, 'Fa': 75, 'TA': 85,'Gd': 95,  'Ex': 110 }}
        self.iterator_list=None
        

    def fit(self,df,**fit_params):
        df_copy=df.copy()
        self.iterator_list=dict()
        for column_name,column_dict in ordinal_categories_dict.items():
            encoder=binary_encoder()
            encoder.fit(df_copy,y=self.dict_of_dicts[column_dict],column=column_name)
            self.iterator_list[column_name]=encoder
        return self

    def transform(self,df,**transform_params):
        if type(self.iterator_list) is type(None):
            raise ValueError('Must Fit before transform')
        df_copy=df.copy()
        for column_name,column_dict in ordinal_categories_dict.items():
            encoder=self.iterator_list[column_name]
            df_copy=encoder.transform(df_copy)
        return df_copy



#Preprocess-data
		
preprocessing=make_pipeline(Null_filler(),dummy_encoder(),ordinal_encoder()) #make_pipeline


processed_X_data=preprocessing.fit_transform(input_data.drop(['SalePrice'],axis=1)) #pre_process inputs
y=input_data_without_Id['SalePrice']



#This section to optimize hyper parameters
'''
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(processed_X_data.drop(['Id'],axis=1).as_matrix(), y.as_matrix(), test_size=0.2)

param_grid = {
    "n_estimators": [1000],
    "learning_rate": [0.01, 0.05 ],
    "max_depth": [3, 5]
}

fit_params = {"eval_set": [(test_X, test_y)], 
              "early_stopping_rounds": 5, 
              "verbose": True} 
model=XGBRegressor()

searchCV = GridSearchCV(model, cv=3, param_grid=param_grid, fit_params=fit_params,n_jobs=4)
search_result=searchCV.fit(train_X, train_y)
'''




from xgboost import XGBRegressor


my_model = XGBRegressor(n_estimators=210,learning_rate=0.05,max_depth=3)
my_model.fit(processed_X_data.drop(['Id'],axis=1).as_matrix(), y.as_matrix())

test_data=pd.read_csv('../input/test.csv')
cleaned_test_data=preprocessing.transform(test_data)
predicted_prices = my_model.predict(cleaned_test_data.drop(['Id'],axis=1).as_matrix())



my_submission = pd.DataFrame({'Id': cleaned_test_data.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)

