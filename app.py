import numpy as np
import pickle
import pandas as pd
# from flasgger import Swagger
import streamlit as st

from PIL import Image
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)


# @app.route('/')
def welcome():
    return "Welcome All"

def predict_in_cost(age, sex, bmi, children, smoker, region):

    prediction = model.predict([[age, sex, bmi, children, smoker, region]])
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
        sex = st.selectbox("Gender: [0: Male, 1: Female]", ["0", "1"], index=0)
        bmi = st.text_input("Bmi [Body Mass Index], Eg:30.90 ", value=23.90)
        children = st.selectbox("No Of Childrens", ["0", "1", "2", "3", "4","5","6","7","8","9","10"])
        smoker = st.selectbox("Are You Smoker? [0 : No, 1: Yes]", ["0", "1"], index=0)
        region = st.selectbox("**Region** : **South_West**--:**1**, **South_East**--:**2**, **North_West**--:**3**, **North_East--**:**4**", ["1", "2", "3", "4"], index=0)
        result = ""

        if st.button("Predict"):
            result =predict_in_cost(age, sex, bmi, children, smoker, region)
        st.success('The output is {}'.format(result))

    if rad == "About Us":
        if st.button("About"):
            st.text("Author : Sooraj ")
            st.text("model used: RandomForestRegression")
            st.text("Built with Streamlit")
            st.text("This project is performed under hamoye data science internship")
            

if __name__ == '__main__':
    main()
