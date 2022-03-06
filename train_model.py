from turtle import pd
import pandas as pd
import numpy as np

## Testing out Random Forest and Boosting classifiers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


df = pd.read_csv('processed_data.csv')
print(df.head())
## Train Test Split 
X = df.drop(['CLASS'], axis=1)
y = df.drop(['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'Gender'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)



### RAndom Forest
model = RandomForestClassifier(bootstrap = False,n_estimators=100 , max_depth=None)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(y_pred_train.shape,y_pred_test.shape)
print(model.score(X_test,y_test))
print(" TRAINING SCORE : " , accuracy_score(y_pred_train, y_train))
print("   TEST SCORE : " ,accuracy_score(y_pred_test, y_test))
# Output
#  TRAINING SCORE :  1.0
#    TEST SCORE :  0.99


# Saving the model 
pickle.dump(model, open("model_diabetes.pkl", 'wb'))




plot_confusion_matrix(model,X_test,y_test)
#plt.show()
plt.savefig('model_diabetes.png')

plot_confusion_matrix(model,X_train,y_train)
#plt.show()
plt.savefig('model_diabetes_trainset.png')
