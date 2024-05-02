#import the libraries
import math
import numpy as np
import pickle
import streamlit as st

#set page width
st.set_page_config(page_title='IPL_SCORE_PREDICTOR', layout="centered")

#Get the ML model
filename = 'C:/Users/Alok/Downloads/IPLscorepredictor/ml_model.pkl'  # Correct path
model = pickle.load(open(filename, 'rb'))

#title of the page with CSS
st.markdown("<h1 style='text-align: center; color: black;'>IPL SCORE PREDICTOR 2024 </h1>", unsafe_allow_html=True)

#Add background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("IPL.jpeg");
        background-attachment: fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Adding description
with st.expander("Description"):
    st.info("""A simple ML model to predict IPL Score between teams in an ongoing match. Here, to make sure the model results more accurate, the minimum number of overs needed to be considered is greater than 5 overs.""")

#Select the Batting Team
batting_team = st.selectbox('Select the Batting Team', ('Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab', 'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'))

prediction_array = []

#Batting Team
if batting_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
elif batting_team == 'Delhi Capitals':
    prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
elif batting_team == 'Kings XI Punjab':
    prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
elif batting_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
elif batting_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
elif batting_team == 'Rajasthan Royals':
    prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
elif batting_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
elif batting_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]

#Select Bowling Team
bowling_team = st.selectbox('Select the Bowling Team', ('Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab', 'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'))

if bowling_team == batting_team:
    st.error('Bowling and Batting teams should be different')

#Bowling Team
if bowling_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1, 0, 0, 0, 0, 0, 0, 0]
elif bowling_team == 'Delhi Capitals':
    prediction_array = prediction_array + [0, 1, 0, 0, 0, 0, 0, 0]
elif bowling_team == 'Kings XI Punjab':
    prediction_array = prediction_array + [0, 0, 1, 0, 0, 0, 0, 0]
elif bowling_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0, 0, 0, 1, 0, 0, 0, 0]
elif bowling_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0, 0, 0, 0, 1, 0, 0, 0]
elif bowling_team == 'Rajasthan Royals':
    prediction_array = prediction_array + [0, 0, 0, 0, 0, 1, 0, 0]
elif bowling_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 1, 0]
elif bowling_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0, 0, 0, 0, 0, 0, 0, 1]

col1, col2 = st.columns(2)

#Enter the current Ongoing Over
with col1:
    overs = st.number_input('Enter the Current Over ', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
    if overs - math.floor(overs) > 0.5:
        st.error('Please enter a valid over input as one over only contains 6 balls')

with col2:
    #Enter Current Run
    runs = st.number_input('Enter Current Runs', min_value=0, max_value=354, step=1, format='%i')

#Wickets taken till now
wickets = st.slider('Enter wickets fallen till now ', 0, 9)

col3, col4 = st.columns(2)

with col3:
    #Runs in last 5 over
    runs_in_prev_5 = st.number_input('Runs scored in last 5 overs ', min_value=0, max_value=runs, step=1, format='%i')

with col4:
    #Wickets in last 5 over
    wickets_in_prev_5 = st.number_input('Wickets fallen in last 5 overs ', min_value=0, max_value=wickets, step=1, format='%i')

#get all the data for predicting
prediction_array = prediction_array + [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]
prediction_array = np.array([prediction_array])
predict = model.predict(prediction_array)

if st.button('Predict Score'):
    #Calling ML model
    my_prediction = int(round(predict[0]))

    #Display the predicted score
    x = f'Predicted match score: {my_prediction + 10} to {my_prediction + 20}'
    st.success(x)
