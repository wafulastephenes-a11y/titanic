import streamlit as st
import pandas as pd
import joblib

#loading model and scaler
model=joblib.load('model.pkl')
scaler=joblib.load('scaler.pkl')
names=joblib.load('names.pkl')
sex_encoder=joblib.load('encoder.pkl')

st.title('Titanic survival prediction')
st.write('Enter details of the passenger')

pclass=st.selectbox('Passenger Class(1=1st,2=2nd,3=3rd)', [1,2,3])
sex=st.selectbox('Sex',sex_encoder.classes_)
age=st.number_input('age',0,100,30)
fare=st.number_input('fare',0.0,600.0,40.0)
embarked=st.selectbox('Portion of embark',['Cherbourg','Queenstown','Southampton'])
alone=st.selectbox('Alone (1= Yes; 0=No)', [0,1])

sex_val=sex_encoder.transform([sex])[0]
embarked_map={'Cherbourg':(0,0),
            'Queenstown':(1,0),
            'Southampton':(2,0)}

embarked_Q,embarked_S=embarked_map[embarked]
input_data=pd.DataFrame([{
    'pclass':pclass,
    'sex':sex_val,
    'age':age,
    'fare':fare,
    'embarked_Q':embarked_Q,
    'embarked_S':embarked_S,
    'alone':alone
}],columns=names)

if st.button('Predict Survival'):
    input_data_scaled=scaler.transform(input_data)
    prediction=model.predict(input_data_scaled)
    result='Survived' if prediction[0]==1 else 'did not survive'
    st.success(f'The passenger would have: {result}')

