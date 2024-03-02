import pandas as pd
import streamlit as st

st.title("Airbnb Price Optimizer")
st.subheader("Airbnb Price Optimizer")


st.markdown(
    """
    <style>
    {
        font-family: 'Inter'
    }
   </style>
    """,
    unsafe_allow_html=True
)

with st.form("form1", clear_on_submit= True): 

    name = st.text_input("Enter full name")
    email = st.text_input("Enter email")

    neighborhood_group = st.selectbox(
        'Where is your listing?',
        ('Brooklyn','Manhattan','Bronx','Queens'))

    st.write('You selected:', neighborhood_group)

    room_type = st.selectbox(
        'What type of room?',
        ('Entire Home/Apartment', 'Private Room', 'Shared Room', 'Hotel Room'))

    st.write('You selected:', room_type)

    # min 1, max 89

    min_nights, max_nights = st.select_slider(
        'Select a range of nights',
        options = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89']
    ,
        value=('1', '89'))
    # st.write('You selected values between', min_nights, 'and', max_nights)

    submit = st.form_submit_button("Submit")

