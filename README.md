# Random-Forest-Diabetes-classifier

 Diabetes Classifier 
=====================

Results obtained: 99% test accuracy and 100% train accuracy
MOdel :Random Forest

Folder Structure
=================
Results : contains model file and confusion matrix(Can estimate number of errors in each class), we have very less prediction errors in test set
Data: data.csv is the original one, processed_data.csv is after converting it to proper form 

Description:
=============
1.I have processed the given data: correct discrepancies in Gender and Class Column, Encoding it , and dropped ID and No ofpatient as part of feature Engineering 
2. Trained a Random Forest model : Parameters choice: n_estimators(the more, the  better accuracy at a tradeoff of model-size), if I take 20,40, the acc ll be slightly less but the model ll be a lot smaller.
This is the number of trees we want to build before taking the maximum voting or averages of predictions. more number of trees give sus better performance but makes your code slower(and larger model)
3.Visulalising results: as confusion matrix saved in pngs and saved the model file
