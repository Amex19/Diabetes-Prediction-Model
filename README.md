### 1. Importing Dependencies 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

### Data Collection and Analysis
### Diabets Dataset

### 2. Loading the Dataset

df=pd.read_csv(r"C:\Users\amare\OneDrive\Desktop\Machine Learning Train a dataset\Datasets\Multiple Disease prediction\diabetes.csv")

df.head()

# number of rows and Columns in this dataset
df.shape

# Getting the summary of Data
pd.options.display.float_format = "{:.2f}".format
df.describe()# for numeric columns

df['Outcome'].value_counts()

0==Non diabetic
1==Diabetic

df.groupby('Outcome').mean()

# separating the data and labels
X = df.drop(columns = 'Outcome', axis=1)
Y = df['Outcome']

print(X)

print(Y)

### Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

### Training the Model

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

### Model Evaluation
### Model Accuracy

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

### Making a Predictive System

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

### Saving the trained model

import pickle # saving the trained model

filename='.trained_model.sav'
pickle.dump(classifier, open(filename,"wb")) # wb=writing binary 

# loading the saved nodel
loaded_model=pickle.load(open('trained_model.sav','rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

## web app developement using Spider-Python IDE

# Diabetes-Prediction-Model # 
import numpy as np
import pickle
import  streamlit as st

loaded_model=pickle.load(open('C:/Users/amare/OneDrive/Desktop/Machine Learning Train a dataset/Deploying Machine Learning Model/trained_model.sav','rb'))


# Creating a function for prediction

def Diabetes_prediction(input_data):

   # changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)

   # reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

   prediction = loaded_model.predict(input_data_reshaped)
   print(prediction)

   if (prediction[0] == 0):
    return 'The person is not diabetic'
   else:
    return 'The person is diabetic'

def main():
    
# Giving a Title

    st.title("Diabetes Prediction Web App")

# Getting the Input Data from the User

    Pregnancies=st.text_input('number of pregnancies') 
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Value') 
    SkinThickness=st.text_input('Skin Thickness Value') 
    Insulin=st.text_input('Insulin Level') 
    BMI=st.text_input('BMI Value') 
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Value') 
    Age=st.text_input('Age of the Person') 
   
   # Code for Prediction 
    Diagnosis=''
   
   #Creating a Button for Prediction 
   
    if st.button('Diabates Test Result'):
        
        Diagnosis=Diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]) 
     
    
    st.success(Diagnosis)     
     
     
if __name__=='__main__':
    main() 
    
