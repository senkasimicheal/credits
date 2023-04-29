from flask import Flask, render_template, request
import io
import base64
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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

@app.route('/plot')
def plot1():
    df = pd.read_csv("customer_data.csv")
    
    credit=df.credit_history.value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(credit,labels=credit.index,autopct='%1.1f%%',startangle=90)
    plt.title("Chart for credit history")
    
    fig2, ax2 = plt.subplots()
    ax2 = sns.barplot(x=df.gender,y=df.credit_amount,ci=None)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                     textcoords='offset points')
    plt.title("Chart for credit amount with gender")
    plt.ylabel("average credit amount")
    
    fig3, ax3 = plt.subplots()
    ax3 = sns.barplot(x=df.purpose, y=df.credit_amount,ci=None)
    for p in ax3.patches:
        ax3.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                     textcoords='offset points')
    plt.xticks(rotation=45)
    plt.xlabel('purpose of credit')
    plt.ylabel('average credit amount')
    plt.title('Chart showing purpose of credit with the average credit amount')
    
    fig4, ax4 = plt.subplots()
    ax4 = sns.scatterplot(df.credit_amount,df.duration)
    ax4 = sns.regplot(df.credit_amount,df.duration)
    plt.xlabel("credit amount")
    plt.ylabel("credit duration")
    plt.title("Linear relationship between credit amount and duration")
    
    buffer1 = io.BytesIO()
    fig1.savefig(buffer1, format='png')
    buffer1.seek(0)
    plot1_png=base64.b64encode(buffer1.getvalue()).decode('utf-8')
    buffer1.close()
    
    buffer2 = io.BytesIO()
    fig2.savefig(buffer2, format='png')
    buffer2.seek(0)
    plot2_png=base64.b64encode(buffer2.getvalue()).decode('utf-8')
    buffer2.close()
    
    buffer3 = io.BytesIO()
    fig3.savefig(buffer3,format='png')
    buffer3.seek(0)
    plot3_png=base64.b64encode(buffer3.getvalue()).decode('utf-8')
    buffer3.close()
    
    buffer4 = io.BytesIO()
    fig4.savefig(buffer4,format='png')
    buffer4.seek(0)
    plot4_png=base64.b64encode(buffer4.getvalue()).decode('utf-8')
    buffer4.close()
    
    return render_template('plot.html',image1=plot1_png,image2=plot2_png,image3=plot3_png,image4=plot4_png)

@app.route('/plot')
def gotoVisuals():
    return render_template('plot.html')

@app.route('/')
def gotoHome():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)