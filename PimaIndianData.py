#####Pima Indian Dibetes Database########
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
### Read dibetise dataset##
#PregnanciesNumber of times pregnant
##GlucosePlasma glucose concentration a 2 hours in an oral glucose tolerance test
#BloodPressureDiastolic blood pressure (mm Hg)
#SkinThicknessTriceps skin fold thickness (mm)
#Insulin2-Hour serum insulin (mu U/ml)
##BMIBody mass index (weight in kg/(height in m)^2)
#DiabetesPedigreeFunctionDiabetes pedigree function
#AgeAge (years)

pima = pd.read_csv("D:\\kagle Data\\diabetes.csv")
print(len(pima))
print(pima.head())

print(pima.describe())
corr=pima.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr , annot=True, cmap="Blues")
plt.title('heatmap',fontsize=15)
####Split The Dataset##
x=pima.iloc[:,0:8]
y=pima.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
### Imoport logistic regression#####
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

y_pred = logmodel.predict(x_test)
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)

print(cnf_matrix)
print('accuracy',metrics.accuracy_score(y_test, y_pred))
y_pred_prob=logmodel.predict_proba(x_test)[::,1]
fpr,tpr,_=metrics.roc_curve(y_test,y_pred_prob)
auc=metrics.roc_auc_score(y_test,y_pred_prob)
plt.plot(fpr,tpr,label="data 1 ,auc"+str(auc))
plt.legend(loc=1)
plt.show



