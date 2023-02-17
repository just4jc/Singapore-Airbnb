import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
from joblib import load
import pickle
from sklearn import preprocessing

st.set_page_config(page_title='Singapore Airbnb Price Predictor', page_icon=':money_with_wings:')

@st.cache_data
def load_data():
    # First load the original airbnb listtings dataset
    data = pd.read_csv("listings.csv") #use this for the original dataset, before transformations and cleaning
    return data

@st.cache_data
def load_vis_data():
    df = pd.read_csv('listings_new2.csv')
    return data


data = load_data()
df = load_vis_data()


# Load the model from P01 Student Chong Xin Le Airbnb section 

st.sidebar.title("Airbnb Singapore Listings: house (room) prices and locations")
st.sidebar.markdown("This web app allows you to explore the Airbnb listings in Singapore. You can filter the listings by a price range between $70-180, neighbourhoods and room type. You can also view the listings on a map in the 'Explore' tab and make predictions in the 'Predict' tab.")

price_range = st.sidebar.slider("Price range (in dollars)", 70, 180, (70, 180))
neighbourhood_group = st.sidebar.selectbox("Neighbourhood group", data['neighbourhood_group'].unique())
room_type = st.sidebar.selectbox("Room type", data['room_type'].unique())

filtered_data = data[(data['price'] >= price_range[0]) & (data['price'] <= price_range[1]) & (data['neighbourhood_group'] == neighbourhood_group) & (data['room_type'] == room_type)]


tab1, tab2 = st.tabs(['Explore', 'Predict'])

with tab1:
    midpoint = (np.average(filtered_data['latitude']), np.average(filtered_data['longitude']))
    st.pydeck_chart(pdk.Deck(

        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=filtered_data[['latitude', 'longitude']],
                get_position='[longitude, latitude]',
                radius=400,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                auto_highlight=True,
                coverage=1,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_data[['latitude', 'longitude']],
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
            # Add another pydeck layer to show only 1 unique neighbourhood name
            #pdk.Layer(
            #    'TextLayer',
            #    data=filtered_data[['latitude', 'longitude', 'neighbourhood']],
            #    get_position='[longitude, latitude]',
            #    get_text='neighbourhood',
            #    get_size=20,
            #    get_color=[310, 0, 200],
            #    get_angle=0,
            #    get_text_anchor='middle',
            #    get_alignment_baseline='center',
            #),
        ],
    ))
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(filtered_data[['neighbourhood', 'price', 'room_type']])

with tab2:
    
    # Define the app title and favicon
    st.title('How Much Can You Make On Airbnb?') 
    st.subheader('Predict')
    st.markdown("This tab allows you to make predictions on the price of a listing based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
    st.write('Choose a neighborhood group, neighborhood, and room type to get the predicted average price.')
    
    with open('rf.pkl', 'rb') as file:
        rf = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Load the cleaned and transformed dataset
    df = pd.read_csv('listings_new2.csv')
    price = df[['price']] # extract price column from listings_new2.csv

    availability_365 = 365
    calculated_host_listings_count = 1
    is_zero = 0
    minimum_nights = 1
    number_of_reviews = 1
    reviews_per_month = 5

    
    # Define the user input functions
    ng_mapping = {'Central Region': 0, 'East Region': 1, 'North Region': 2, 'North-East Region': 3, 'West Region': 4}
    ng_reverse_mapping = {v: k for k, v in ng_mapping.items()}
    ng_labels = list(ng_mapping.keys())
    
    n_mapping = {
        'Bedok': 0,
        'Bishan': 1,
        'Bukit Batok': 2,
        'Bukit Merah': 3,
        'Bukit Panjang': 4,
        'Bukit Timah': 5,
        'Central Area': 6,
        'Choa Chu Kang': 7,
        'Clementi': 8,
        'Geylang': 9,
        'Hougang': 10,
        'Jurong East': 11,
        'Jurong West': 12,
        'Kallang': 13,
        'Mandai': 14,
        'Marine Parade': 15,
        'Novena': 16,
        'Outram': 17,
        'Pasir Ris': 18,
        'Punggol': 19,
        'Queenstown': 20,
        'River Valley': 21,
        'Rochor': 22,
        'Sembawang': 23,
        'Sengkang': 24,
        'Serangoon': 25,
        'Tampines': 26,
        'Tanglin': 27,
        'Toa Payoh': 28,
        'Woodlands': 29,
        'Yishun': 30
    }    
    
    room_type_mapping = {'Private room': 0, 'Entire home/apt': 1}
    room_type_reverse_mapping = {}
    for k, v in room_type_mapping.items():
        room_type_reverse_mapping[v] = k
        #print("Added room type %s with key %s" % (v, k))
    
    ng_labels = [ng_reverse_mapping[i] for i in sorted(ng_reverse_mapping.keys())]
    #n_labels = [n_reverse_mapping[i] for i in sorted(n_reverse_mapping.keys())]

    rt_labels = [room_type_reverse_mapping[i] for i in sorted(room_type_reverse_mapping.keys())]

    
    def get_neighbourhood_group():
        neighbourhood_group = st.selectbox('Select a neighborhood group', ng_labels)
        return neighbourhood_group


    def get_neighbourhood(neighbourhood_group):
        # show only the neighbourhoods in the selected neighbourhood group
        neighbourhoods = df[df['neighbourhood_group'] == ng_mapping[neighbourhood_group]]['neighbourhood'].unique()
        # map n_mapping to a categorical value based off of the selected neighbourhood group using the n_mapping dictionary key value pairs
        #n_mapping = {i: neighbourhood for i, neighbourhood in enumerate(neighbourhoods)}
        #n_mapping = {neighbourhood: i for i, neighbourhood in enumerate(neighbourhoods)}
        # show the neighbourhoods in categorical order based off of the selected neighbourhood group

        neighbourhood = st.selectbox('Select a neighbourhood for the neighbourhood group', n_mapping)
        #neighbourhood = st.selectbox('Select a neighbourhood for the neighbourhood group', list(n_mapping.keys()), format_func=lambda x: n_mapping[x])
        #return n_mapping[neighbourhood]
        return neighbourhood
    
    def get_room_type():
        #room_type = st.selectbox('Select a room type', df['room_type'].unique())
        room_type = st.selectbox('Select a room type', rt_labels)
        return room_type


    # Define the user input fields
    ng_input = get_neighbourhood_group()
    n_input = get_neighbourhood(ng_input)
    room_type_input = get_room_type()

    # Map user inputs to integer encoding
    ng_int = ng_mapping[ng_input]
    n_int = n_mapping[n_input]
    rt_int = room_type_mapping[room_type_input]
    
    # Display the prediction
    if st.button('Predict Price'):
        
        # Make the prediction   
        input_data = [[minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_int, n_int, rt_int, reviews_per_month, is_zero]]
        input_df = pd.DataFrame(input_data, columns=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count','availability_365','neighbourhood_group', 'neighbourhood', 'room_type', 'reviews_per_month', 'is_zero'])
        prediction = rf.predict(input_df)   
        # convert output data and columns, including price, to a dataframe avoiding TypeError: type numpy.ndarray doesn't define __round__ method
        output_data = [minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_input, n_input, rt_int, reviews_per_month, is_zero, prediction[0]]

    
        output_df = pd.DataFrame([output_data], columns=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count','availability_365','neighbourhood_group', 'neighbourhood', 'room_type', 'reviews_per_month', 'is_zero', 'predicted_price'])

        # Make the prediction   
        # show prediction on price in dollars and cents using the price column 
        input_data = [[minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_int, n_int, rt_int, reviews_per_month, is_zero]]

        predicted_price = rf.predict(input_df)[0]
        st.write('The predicted average price is ${:.2f}.'.format(predicted_price))
        st.dataframe(output_df)