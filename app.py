from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
model2=pickle.load(open('model2.pkl','rb'))
@app.route("/")
def hello_world():
    return render_template('index.html')
@app.route("/app1239/diabetes")
def hello_world2():
    return render_template('diabetes.html')
@app.route("/app1239/heart")
def hello_world3():
    return render_template('heart.html')


@app.route("/app1239/predict",methods=['POST','GET'])
def predict():
    int_features =[x for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    if prediction==0:
        return render_template('heart.html',pred='you are heart patient')
    else:
        return render_template('heart.html',pred='you are not heart patient')
@app.route("/app1239/predict2",methods=['POST','GET'])
def predict2():
    int_features2 =[x for x in request.form.values()]
    final2=[np.array(int_features2)]
    print(int_features2)
    print(final2)
    prediction=model2.predict(final2)
    if prediction==0:
        return render_template('diabetes.html',pred='you are diabetes patient')
    else:
        return render_template('diabetes.html',pred='you are not diabetes patient')



if __name__ == "__main__":
    app.run(debug=True,port=8014 )
