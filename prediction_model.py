import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import pickle
data = pd.read_csv('heart.csv')
data.head()
data.tail()
data.shape
print("no of rows",data.shape[0])
print("no of columns",data.shape[1])
data.info()
data.isnull()

data.isnull().sum()
data.describe()
data_copy=data.copy(deep=True)
data.columns
data_copy.isnull().sum()
data_copy[['age']]=data_copy[['age']].replace(0,np.nan)
data_copy[['sex']]=data_copy[['sex']].replace(0,np.nan)
data_copy[['cp']]=data_copy[['cp']].replace(0,np.nan)
data_copy[['trestbps']]=data_copy[['trestbps']].replace(0,np.nan)
data_copy[['chol']]=data_copy[['chol']].replace(0,np.nan)
data_copy[['fbs']]=data_copy[['fbs']].replace(0,np.nan)
data_copy[['restecg']]=data_copy[['restecg']].replace(0,np.nan)
data_copy[['thalach']]=data_copy[['thalach']].replace(0,np.nan)
data_copy[['exang']]=data_copy[['exang']].replace(0,np.nan)
data_copy[['oldpeak']]=data_copy[['oldpeak']].replace(0,np.nan)
data_copy[['slope']]=data_copy[['slope']].replace(0,np.nan)
data_copy[['ca']]=data_copy[['ca']].replace(0,np.nan)
data_copy[['thal']]=data_copy[['thal']].replace(0,np.nan)
data_copy.isnull()
data['age']=data['age'].replace(0,data['age'].mean())
data['sex']=data['sex'].replace(0,data['sex'].mean())
data['cp']=data['cp'].replace(0,data['cp'].mean())
data['trestbps']=data['chol'].replace(0,data['chol'].mean())
data['fbs']=data['fbs'].replace(0,data['fbs'].mean()) 
data['restecg']=data['restecg'].replace(0,data['restecg'].mean())
data['thalach']=data['thalach'].replace(0,data['thalach'].mean())
data['exang']=data['exang'].replace(0,data['exang'].mean())
data['oldpeak']=data['oldpeak'].replace(0,data['oldpeak'].mean())
data['slope']=data['slope'].replace(0,data['slope'].mean()) 
data['ca']=data['ca'].replace(0,data['ca'].mean())
data['thal']=data['thal'].replace(0,data['thal'].mean())
x = data.drop('target',axis=1)
y = data['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
pipeline_lr = Pipeline([('scalar1',StandardScaler()),('lr_classifier',LogisticRegression())])
pipeline_knn = Pipeline([('scalar2',StandardScaler()),('knn_classifier',KNeighborsClassifier())])
pipeline_svc = Pipeline([('scalar3',StandardScaler()),('svc_classifier',SVC())])
pipeline_dt = Pipeline([('dt_classifier',DecisionTreeClassifier())])
pipeline_rf = Pipeline([('rf_classifier',RandomForestClassifier())])
pipeline_gbc = Pipeline([('gbc_classifier',GradientBoostingClassifier())])
pipelines = [pipeline_lr ,pipeline_knn,pipeline_svc,pipeline_dt,pipeline_rf,pipeline_gbc]
for pipe in pipelines:
    pipe.fit(x_train,y_train)
pipe_dict ={0:'LR',1:'KNN',2:'SVC',3:'DT',4:'RF',5:'GBC'}
pipe_dict
for i,model in enumerate(pipelines):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(x_test,y_test)*100))
    x = data.drop('target',axis=1)
y = data['target']
DT = DecisionTreeClassifier()
DT.fit(x,y)
DecisionTreeClassifier()
new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
    'slope':2,
    'ca':2,
    'thal':3,
},index=[0])
p=DT.predict(new_data)
if(p[0]==0):
    print("no heart disease detected")
else:
    print("heart disease detected")
pickle.dump(DT,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))