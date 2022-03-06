import csv
import pandas as pd
import os
import numpy as np

## read teh diabetes CSV
df = pd.read_csv('data.csv')
#print(df.head())

## removing columns that are not necessary: ID , and No_pation
df = df.drop(['ID','No_Pation'],axis = 1)
#print(df.head())

# print(df['Gender'].unique())
# print(df['CLASS'].unique())
# ['F' 'M' 'f']
# ['N' 'N ' 'P' 'Y' 'Y ']
# SO first should replace f with F, 'N 'with 'N' and 'P ' with P
df['Gender'] = df['Gender'].replace('f' , 'F')
df['CLASS'] = df['CLASS'].replace('Y ' , 'Y')
df['CLASS'] = df['CLASS'].replace('N ' , 'N')
# print(df.head())
# print(df['Gender'].unique())
# print(df['CLASS'].unique())
# ['F' 'M']
# ['N' 'P' 'Y']
# Corrected Now


## Checking if Nan is present
print(df.isnull().values.any())
## No Nan Values    
df.fillna(0)



## ## encoding for GEnder and Class(Our TArget variable) as Strings are not parsed in ML models
df['CLASS'] = df['CLASS'].replace('N' , 0)
df['CLASS'] = df['CLASS'].replace('P' , 1)
df['CLASS'] = df['CLASS'].replace('Y' , 2)
df['Gender'] = df['Gender'].replace('M' , 0)
df['Gender'] = df['Gender'].replace('F' , 1)

print(df.head())
df.to_csv('processed_data.csv')



