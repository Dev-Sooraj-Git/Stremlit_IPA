import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)
df = pd.read_csv("insurance.csv")
df['sex_con'] = df['sex'].apply({'male':0, 'female':1}.get)
Gender_dict = dict(zip(df['sex'], df['sex_con']))
df['smoker_con'] = df['smoker'].apply({'yes':1, 'no':0}.get)
Smoker_dict = dict(zip(df['smoker'], df['smoker_con']))
df['region_con'] = df['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)
Region_dict = dict(zip(df['region'], df['region_con']))

# @app.route('/')
def welcome():
    return "Welcome All"

def predict_in_cost(age, Gender, bmi, children, Smoker, Region):

    prediction = model.predict([[age, Gender, bmi, children, Smoker, Region]])
    print(prediction)
    return prediction

rad =st.sidebar.radio("Navigation",["Home","About Us"])
def main():
    st.title("Insurance Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">HEALTH INSURANCE CHARGES PREDICTION ML APP </h2>
    </div>
    <marquee> Welcome To Insurance Prediction, Have a Good Day </marquee>
    """

    if rad == "Home":
        st.markdown(html_temp, unsafe_allow_html=True)
        age = st.text_input("Age, Eg : 23", value=19)
        sex = st.selectbox("Select Gender:", df['sex'].unique())
        #sex = st.selectbox("Gender: [0: Male, 1: Female]", ["0", "1"], index=0)
        bmi = st.text_input("Bmi [Body Mass Index], Eg:30.90 ", value=23.90)
        children = st.selectbox("No Of Childrens", ["0", "1", "2", "3", "4","5","6","7","8","9","10"])
        smoker = st.selectbox("Are You Smoker?",df['smoker'].unique())
        region = st.selectbox("Region:", df['region'].unique())
        result = ""

        if st.button("Predict"):
            Gender = Gender_dict[sex]
            Smoker = Smoker_dict[smoker]
            Region = Region_dict[region]
            result =predict_in_cost(age, Gender, bmi, children, Smoker, Region)
        st.success('The output is {}'.format(result))

    if rad == "About Us":
        if st.button("About"):
            st.text("Author : Sooraj S")
            st.text("model used: RandomForestRegression")
            st.text("Built with Streamlit")
            st.text("This project is performed under hamoye data science internship")
            

if __name__ == '__main__':
    main()