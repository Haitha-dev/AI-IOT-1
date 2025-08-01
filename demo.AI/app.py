import pandas as pd#import the pandas to read csv file
from sklearn.preprocessing import LabelEncoder#to covert output non numerical to numerical
from sklearn.model_selection import train_test_split#to trai and  test and split
from sklearn.linear_model import LogisticRegression#it is for input
import joblib

df=pd.read_csv('feeds.csv')
label_encoder=LabelEncoder()#by using this to covert the non num to numerical data 
df['action']=label_encoder.fit_transform(df['action'])#it is coverting data into binaryform by using fit transform

X=df[['temperature','humidity']]#by using label data splitting the columns#two coloums data inserting in x variable
Y=df[['action']]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,Y_train)
#download the model formate file for that install joblib
joblib.dump(model,'ac_model.pkl')

print("model trained downloaded sucessfully")



