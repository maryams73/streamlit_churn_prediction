import streamlit as st
import numpy as np
import pandas as pd
import joblib


model = joblib.load('finalized_model.sav')
st.title('Churn Prediction using customer information:walking:')

#define conversion functions
def to_dummies(array, value):
    index = array.index(value)
    arr = [0] * len(array)
    arr[index] = 1
    return arr

def to_binary(val): 
    if (val=='Yes'):
        return 1  
    else:
        return 0
    
def minmax_scaler(x,min, max):
    return (x-min)/(max-min)   

internetServiceValues = ['DSL', 'Fiber optic', 'No'] 
contractValues = ['Month-to-month', 'One year', 'Two year']
paymentMethodValues = ['Bank transfer (automatic)', 'Credit card (automatic)','Electronic check', 'Mailed check']
columns_final = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                    'InternetService_DSL', 'InternetService_Fiber optic',
                    'InternetService_No', 'Contract_Month-to-month', 'Contract_One year',
                    'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
                    'PaymentMethod_Credit card (automatic)',
                    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'] 

# User Input
gender = st.radio("Gender: ", ['Male','Female'])
seniorCitizen = st.radio("Senior Citizen: ", ['Yes','No'])
partner = st.radio("Partner: ", ['Yes','No'])
dependents = st.radio("Dependents: ", ['Yes','No'])
tenure = int(st.slider("Choose tenure: ",1,72))
phoneService = st.radio("PhoneService: ", ['Yes','No'])
multipleLines = st.radio("Multiple Lines: ", ['Yes','No'])
onlineSecurity = st.selectbox("Online Security: ", ['No', 'Yes', 'No internet service'])
onlineBackup = st.selectbox("Online Backup: ", ['Yes', 'No', 'No internet service'])
deviceProtection = st.selectbox("Device Protection: ", ['No', 'Yes', 'No internet service'])
techSupport = st.selectbox("Tech Support: ", ['No', 'Yes', 'No internet service'])
streamingTV = st.selectbox("Streaming TV: ", ['No', 'Yes', 'No internet service'])
streamingMovies = st.selectbox("Streaming Movies: ", ['No', 'Yes', 'No internet service'])
paperlessBilling = st.radio("Paperless Billing: ", ['Yes','No'])
monthlyCharges = st.slider("Monthly Charges: ", min_value=18.00, max_value=120.00, step=0.05, format="%.2f")
totalCharges = st.slider("Total Charges: ", min_value=18.00, max_value=8700.00, step=0.05, format="%.2f")
internetService = st.selectbox("Internet Service: ", ['DSL','Fiber optic','No'])
contract = st.selectbox("Contract: ", ['Month-to-month', 'One year', 'Two year'])
paymentMethod = st.selectbox("Contract: ", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])



def predict(): 
    #convert to number
    global gender, seniorCitizen, partner, dependents,tenure, phoneService, multipleLines, internetService, onlineSecurity, onlineBackup, deviceProtection,techSupport, streamingTV, streamingMovies, contract, paperlessBilling, paymentMethod, monthlyCharges, totalCharges, scaler

    gender = int(gender == 'Male') 
    seniorCitizen = int(seniorCitizen == 'Yes') 
    partner = int(partner == 'Yes') 
    dependents = int(dependents == 'Yes') 
    tenure = minmax_scaler(tenure, 1, 72)
    phoneService = int(phoneService == 'Yes') 
    multipleLines = int(multipleLines == 'Yes') 
    onlineSecurity = to_binary(onlineSecurity)
    onlineBackup = to_binary(onlineBackup)
    deviceProtection = to_binary(deviceProtection)
    techSupport = to_binary(techSupport)
    streamingTV = to_binary(streamingTV)
    streamingMovies = to_binary(streamingMovies)
    paperlessBilling = int(paperlessBilling == 'Yes') 
    monthlyCharges = minmax_scaler(monthlyCharges, 18.00, 120.00)
    totalCharges= minmax_scaler(totalCharges, 18.00, 8700.00)
    internetService = to_dummies(internetServiceValues, internetService) #needs append
    contract = to_dummies(contractValues, contract) #needs append
    paymentMethod = to_dummies(paymentMethodValues, paymentMethod) #needs append
    
    #making dataframe
    row = np.array([gender, seniorCitizen, partner, dependents,tenure, phoneService, multipleLines, onlineSecurity, onlineBackup, 
                    deviceProtection, techSupport, streamingTV, streamingMovies, paperlessBilling,monthlyCharges, totalCharges])
    row = np.append(row, internetService)
    row = np.append(row, contract)
    row = np.append(row,paymentMethod)
    #print("rows: ", row)
    
    X = pd.DataFrame([row], columns=columns_final)
    #print("dataframe1: ",X)

    # using dictionary to convert specific columns
    convert_dict = {'gender': int,
                    'SeniorCitizen': int,
                    'Partner': int, 
                    'Dependents': int, 
                    'tenure': float,
                    'PhoneService': int, 
                    'MultipleLines': int, 
                    'OnlineSecurity': int, 
                    'OnlineBackup': int,
                    'DeviceProtection': int, 
                    'TechSupport': int, 
                    'StreamingTV': int, 
                    'StreamingMovies': int,
                    'PaperlessBilling': int, 
                    'MonthlyCharges': float, 
                    'TotalCharges': float,
                    'InternetService_DSL': int, 
                    'InternetService_Fiber optic': int,
                    'InternetService_No': int, 
                    'Contract_Month-to-month': int, 
                    'Contract_One year': int,
                    'Contract_Two year': int, 
                    'PaymentMethod_Bank transfer (automatic)': int,
                    'PaymentMethod_Credit card (automatic)': int,
                    'PaymentMethod_Electronic check': int, 
                    'PaymentMethod_Mailed check': int
                    }
 
    df = X.astype(convert_dict)
    #print(df.dtypes)
    #print("dataframe2: ",df)

    #make prediction
    prediction = model.predict(df)

    #show the result
    if prediction[0] == 0: 
        st.success('Customer will stay! :thumbsup:')
    else: 
        st.error('Customer will leave! :thumbsdown:') 

    

trigger = st.button('Predict', on_click=predict)


#Key: value > Transformation
#gender: ['Female' 'Male']
#SeniorCitizen: [0 1]
#Partner: ['Yes' 'No'] > [1 0]
#Dependents: ['No' 'Yes'] > [0 1]
#tenure: [ 1 34  2 45  8 22 10 28 62 13 16 58 49 25 69 52 71 21 12 30 47 72 17 27
#  5 46 11 70 63 43 15 60 18 66  9  3 31 50 64 56  7 42 35 48 29 65 38 68
# 32 55 37 36 41  6  4 33 67 23 57 61 14 20 53 40 59 24 44 19 54 51 26 39] 1>>>72
#PhoneService: ['No' 'Yes'] > [0 1]
#MultipleLines: ['No phone service' 'No' 'Yes'] > [0 1]
#InternetService: ['DSL' 'Fiber optic' 'No'] 
#OnlineSecurity: ['No' 'Yes' 'No internet service'] > [0 1]
#OnlineBackup: ['Yes' 'No' 'No internet service'] > [1 0]
#DeviceProtection: ['No' 'Yes' 'No internet service'] > [0 1]
#TechSupport: ['No' 'Yes' 'No internet service'] > [0 1]
#StreamingTV: ['No' 'Yes' 'No internet service'] > [0 1]
#StreamingMovies: ['No' 'Yes' 'No internet service'] > [0 1]
#Contract: ['Month-to-month' 'One year' 'Two year']
#PaperlessBilling: ['Yes' 'No'] > [1 0]
#PaymentMethod: ['Electronic check' 'Mailed check' 'Bank transfer (automatic)' 'Credit card (automatic)']
#MonthlyCharges: [29.85 56.95 53.85 ... 63.1  44.2  78.7 ]
#TotalCharges: [  29.85 1889.5   108.15 ...  346.45  306.6  6844.5 ]
#Churn: ['No' 'Yes']