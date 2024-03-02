import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))

st.sidebar.title("Bankruptcy Prediction using Logistic Regression")

list_of_values = [0.0, 0.5, 1.0]

industrial_risk = st.sidebar.selectbox('Industrial Risk', list_of_values)# Give Label and Options
management_risk = st.sidebar.selectbox('Management Risk', list_of_values)
financial_flexibility = st.sidebar.selectbox('Financial Flexibility', list_of_values)
credibility = st.sidebar.selectbox('Credibility', list_of_values)
competitiveness = st.sidebar.selectbox('Competitiveness', list_of_values)
operating_risk = st.sidebar.selectbox('Operating Risk', list_of_values)

inputs = [industrial_risk, management_risk, financial_flexibility, credibility,
                   competitiveness, operating_risk]

st.write("""
         The data set includes the following variables:

1.	Industrial Risk: 0 =low risk, 0.5 =medium risk, 1 =high risk.
2.	Management Risk: 0 =low risk, 0.5 =medium risk, 1 =high risk.
3.	Financial Flexibility: 0 =low flexibility, 0.5 =medium flexibility, 1 =high flexibility.
4.	Credibility: 0 =low credibility, 0.5 =medium credibility, 1 =high credibility.
5.	Competitiveness: 0 =low competitiveness, 0.5 =medium competitiveness, 1 =high competitiveness.
6.	Operating Risk: 0 =low risk, 0.5 =medium risk, 1 =high risk.
         """)

st.header("Values Entered are as follows: ")

st.write(pd.DataFrame(inputs, index=['Industrial risk', 'Management risk', 'Financial flexibility', 'Credibility',
                   'Competitiveness', 'Operating Risk'], columns=['Values']))

predict = st.sidebar.button('Predict')

if predict:
    inputs = [[industrial_risk, management_risk, financial_flexibility, credibility,
               competitiveness, operating_risk]]  # two brackets as it has to be 2D array

    prediction = model.predict(inputs)
    probability = model.predict_proba(inputs)
    st.header('The probability value is as shown')
    st.write(probability)

    # Display

    if prediction == 0:
        st.header('The Company is likely to be Bankrupt')
    else:
        st.header('The Company is likely to be Non-Bankrupt')















