import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.pipeline import Pipeline
data = pd.read_csv('diabetes.csv')
data.head()
data.tail()
data.shape
print("no of rows",data.shape[0])
print("no of columns",data.shape[1])
data.info()
data.isnull()
data.isnull().sum()
data.describe()
import numpy as np
data_copy=data.copy(deep=True)
data.columns
data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']]=data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']].replace(0,np.nan)
data_copy.isnull().sum()
data_copy[['Pregnancies']]=data_copy[['Pregnancies']].replace(0,np.nan)
data_copy[['Glucose']]=data_copy[['Glucose']].replace(0,np.nan)
data_copy[['BloodPressure']]=data_copy[['BloodPressure']].replace(0,np.nan)
data_copy[['SkinThickness']]=data_copy[['SkinThickness']].replace(0,np.nan)
data_copy[['Insulin']]=data_copy[['Insulin']].replace(0,np.nan)
data_copy[['BMI']]=data_copy[['BMI']].replace(0,np.nan)
data_copy[['DiabetesPedigreeFunction']]=data_copy[['DiabetesPedigreeFunction']].replace(0,np.nan)
data_copy[['Age']]=data_copy[['Age']].replace(0,np.nan)
data_copy.isnull()
data['Glucose']=data['Glucose'].replace(0,data['Glucose'].mean())
data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].mean())
data['Insulin']=data['Insulin'].replace(0,data['Insulin'].mean())
data['BMI']=data['BMI'].replace(0,data['BMI'].mean())          
x = data.drop('Outcome',axis=1)
y = data['Outcome']
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
    from sklearn.ensemble import RandomForestClassifier
    x = data.drop('Outcome',axis=1)
y = data['Outcome']
rf = RandomForestClassifier(max_depth=3)
rf.fit(x,y)
new_data = pd.DataFrame({
    'Pregnancies':6,
    'Glucose':148.0,
    'BloodPressure':72.0,
    'SkinThickness':35.0,
    'Insulin':79.799479,
    'BMI':33.6,
    'DiabetesPedigreeFunction':0.627,
    'Age':50,
},index=[0])
p=rf.predict(new_data)
if p[0] == 0:
    print("non-diabetic")
else:
          print("diabetic")
pickle.dump(rf,open('model2.pkl','wb'))
model=pickle.load(open('model2.pkl','rb'))