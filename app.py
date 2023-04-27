from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  LabelEncoder

app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv("customer_data.csv")
    encoded = LabelEncoder()
    df['cat_purpose'] = encoded.fit_transform(df.purpose)
    df['cat_telephone'] = encoded.fit_transform(df.own_telephone)
    df['cat_gender'] = encoded.fit_transform(df.gender)
    df['cat_marital_status'] = encoded.fit_transform(df.marital_status)
    df['cat_employment'] = encoded.fit_transform(df.employment) 
    
    y = df.credit_history
    x = df[['cat_telephone', 'cat_gender', 'cat_marital_status', 'cat_employment', 'credit_amount']]
    
    # Creating model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x,y)
    pickle.dump(knn,open("data.pkl","wb"))
    return render_template("index.html")

@app.route("/",methods=["GET","POST"])
def predict():
    credit_amount = request.form['credit_amount']
    form_array = np.array([[0,0,0,0,credit_amount]])
    
    telephone = request.form['telephone']
    if telephone == "Yes":
        form_array[:,0]=1
    elif telephone == "No":
        form_array[:,0]=0
        
    gender = request.form['gender']
    if gender == "Male":
        form_array[:,1]=1
    elif gender == "Female":
        form_array[:,1]=0 
    
    mstatus = request.form['mstatus']
    if mstatus == "single":
        form_array[:,2]=2
    elif mstatus == "married":
        form_array[:,2]=1
    elif mstatus == "divorced":
        form_array[:,2]=0
    
    employment = request.form['employment']
    if employment == "less than 1 year experience":
        form_array[:,3]=0
    elif employment == "btn 1 to 4 yrs experiece":
        form_array[:,3]=2
    elif employment == "btn 4 to 7 yrs experiece":
        form_array[:,3]=3
    elif employment == "more than 7 yrs experiece":
        form_array[:,3]=1
    elif employment == "unemployed":
        form_array[:,3]=4
    
    model = pickle.load(open("data.pkl","rb"))
    prediction = model.predict(form_array)[0]
    result = str(prediction)
    eligibility = 'No answer'
    if result == 'critical/other existing credit':
        eligibility = 'Not eligible for a credit'
    elif result == 'existing credit':
        eligibility = 'Might be eligible for a credit'
    elif result == 'delayed previously':
        eligibility = 'Might be eligible for a credit'
    elif result == 'all paid':
        eligibility = 'Eligible for a credit'
        
    return render_template("index.html", result=result, eligibility=eligibility)

if __name__ == "__main__":
    app.run(debug=True)