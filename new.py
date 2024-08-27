from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import requests
import pickle
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title='Singapore Airbnb Price Predictor', page_icon=':money_with_wings:')

@st.cache_data
#@st.cache
def load_data():
    # First load the original airbnb listtings dataset
    data = pd.read_csv("listings.csv") #use this for the original dataset, before transformations and cleaning
    #filter out listings with price = 0 or < 1000 and availability_365 = 0 and number of reviews = 0
    data = data[(data['price'] != 0) & (data['price'] < 1000) & (data['availability_365'] > 0) & (data['number_of_reviews'] != 0) & (data['longitude'] != 0) & (data['latitude'] != 0)]
    return data

@st.cache_data
#@st.cache
def load_vis_data():
    df = pd.read_csv('listings_new2.csv')
    return df


data = load_data()
df = load_vis_data()

# populate neighbourhood group and corresponding neighbourhood dropdowns from neighbourhood.csv
neighbourhood_df = pd.read_csv('neighbourhood.csv')
neighbourhood_df = neighbourhood_df.drop(labels=["Unnamed: 0"], axis=1)

#populate a streamlit web app dropdown with the neighbourhood_group from neighbourhood_df and based on neighbourhood_group selected, populate another dropdown with the neighbourhood from neighbourhood_df based on neighbourhood_group selected


st.sidebar.title("Airbnb Singapore Listings: house (room) prices and locations")
st.sidebar.markdown("This web app allows you to explore the Airbnb listings in Singapore. You can filter the listings by a price range between $70-180, neighbourhoods and room type. You can also view the listings on a map in the 'Explore' tab and make predictions in the 'Predict' tab.")

price_range = st.sidebar.slider("Price range (in dollars)", 70, 180, (70, 180))

if 'price_range' not in st.session_state:
    st.session_state.price_range = (70, 180)
    st.session_state.price_change_flag = True
    
neighbourhood_group = st.sidebar.selectbox("Neighbourhood group", neighbourhood_df['neighbourhood_group'].unique(), key='neighbourhood_group')
neighbourhood_group_list = list(neighbourhood_df.neighbourhood_group.unique())
neighbourhood_group_list.sort()

if 'neighbourhood_group' not in st.session_state:
    st.session_state.neighbourhood_group = neighbourhood_df['neighbourhood_group'].unique()[0]
    st.session_state.neighbourhood_group_change_flag = False


neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhood_df[neighbourhood_df['neighbourhood_group']==neighbourhood_group]['neighbourhood'].unique(), key='neighbourhood')
if 'neighbourhood' not in st.session_state:
    st.session_state.neighbourhood = neighbourhood_df[neighbourhood_df['neighbourhood_group'] == st.session_state.neighbourhood_group]['neighbourhood'].unique()[0]
    st.session_state.neighbourhood_change_flag = False
    
room_type = st.sidebar.selectbox("Room type", data['room_type'].unique(), key='room_type')

    
tab1, tab2 = st.tabs(['Explore', 'Predict'])

with tab1:
    filtered_data = data[(data['price'] >= price_range[0]) & (data['price'] <= price_range[1]) & (data['neighbourhood_group'] == neighbourhood_group) & (data['room_type'] == room_type) & (data.neighbourhood == neighbourhood)]
    if not filtered_data.empty:
        filtered_data_clean = filtered_data[['latitude', 'longitude']].dropna()
        filtered_data_json = filtered_data_clean.to_json(orient='records')
        midpoint = (np.average(filtered_data['latitude']), np.average(filtered_data['longitude']))
        #converting the filtered data to a JSON-serializable format before passing it to st.pydeck_chart
        # Convert the cleaned DataFrame to JSON
            
        #filtered_data_df = pd.read_json(filtered_data_json)
        filtered_data_df = pd.read_json(StringIO(filtered_data_json))
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
                    data=filtered_data_df,
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
                    data=filtered_data_df,
                    get_position='[longitude, latitude]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=200,
                ),
            ],
        ))
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(filtered_data[['neighbourhood', 'price', 'room_type']])

# Tab 2: Predict the price of a listing based on the features, using a trained random forest model
# using neighbourhood_group, neighbourhood, room_type, availability_365, calculated_host_listings_count, minimum_nights, number_of_reviews, reviews_per_month as features
with tab2:
    # Load the serialized trained model rf.pkl and scaler object scaler.pkl
    with open('rf.pkl', 'rb') as file:
        rf = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Load the cleaned and transformed dataset
    df = pd.read_csv('listings_new2.csv')
    price = df[['price']] # extract price column from listings_new2.csv

    # Define the default values for the features
    availability_365 = 365
    calculated_host_listings_count = 1
    is_zero = 0
    minimum_nights = 1
    number_of_reviews = 1
    reviews_per_month = 5
    
    # Define the app title and favicon
    st.title('How Much Can You Make On Airbnb?') 
    st.subheader('Predict')
    st.markdown("This tab allows you to make predictions on the price of a listing based on the region, neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
    st.write('Choose a neighborhood group, neighborhood, and room type to get the predicted average price.')

    # Define a dictionary (n_mapping) that maps neighborhood to their corresponding integer values for each neighborhood group.
    n_mapping = {
        'Central Region': {
            'Geylang': 0,
            'Tanglin': 1,
            'River Valley': 2,
            'Rochor': 3,
            'Outram': 4,
            'Marine Parade': 5,
            'Novena': 6,
            'Queenstown': 7,
            'Toa Payoh': 8,
            'Bukit Merah': 9,
            'Kallang': 10,
            'Bukit Timah': 11,
            'Newton': 12,
            'Downtown Core': 13,
            'Singapore River': 14,
            'Orchard': 15,
            'Bishan': 16,
            'Southern Islands': 17,
            'Museum': 18,
            'Marina South': 19
        },
        'East Region': {
            'Bedok': 0,
            'Pasir Ris': 1,
            'Tampines': 2
        },
        'North Region': {
            'Mandai': 0,
            'Sembawang': 1,
            'Woodlands': 2,
            'Yishun': 3,
            'Central Water Catchment': 4,
            'Lim Chu Kang': 5,
            'Sungei Kadut': 6,
        },
        'North-East Region': {
            'Hougang': 0,
            'Punggol': 1,
            'Sengkang': 2,
            'Serangoon': 3,
            'Ang Mo Kio': 4
        },
        'West Region': {
            'Bukit Batok': 0,
            'Bukit Panjang': 1,
            'Choa Chu Kang': 2,
            'Clementi': 3,
            'Jurong East': 4,
            'Jurong West': 5,
            'Tuas': 6,
            'Western Water Catchment': 7
        }
    }
    
    # map neighbourhood to int using dictionary according to neighbourhood group selection.
    ng_mapping = {'Central Region': 0, 'East Region': 1, 'North Region': 2, 'North-East Region': 3, 'West Region': 4}
    
    # Create a function that takes neighbourhood_group as an argument and returns the corresponding integer value.
    def match_neighbourhood_group(neighbourhood_group):
        return ng_mapping[neighbourhood_group]
    
    # Call function match_neighbourhood_group with the selected neighbourhood_group as an argument to get the corresponding integer value
    ng_int = match_neighbourhood_group(neighbourhood_group)
    
    # Create a function (match_neighbourhood) that takes the 
    # selected neighbourhood_group and neighbourhood as arguments 
    # and returns the corresponding integer value by looking up n_mapping.
    def match_neighbourhood(neighbourhood_group, neighbourhood):
        return n_mapping[neighbourhood_group][neighbourhood]

    # Call function match_neighbourhood with the selected neighbourhood_group and neighbourhood 
    # as arguments to get the corresponding integer value
    n_int = match_neighbourhood(neighbourhood_group, neighbourhood)
    
    # create a dictionary with keys as the room types and values as the corresponding integer number
    # this is to map the room type to an integer value for the model to make a prediction
    room_type_mapping = {'Entire home/apt':0, 'Private room': 1, 'Shared room': 2}
    
    # Create a function (match_room_type) that takes the selected room_type as an argument
    # and returns the corresponding integer value.
    def match_room_type(room_type):
        return room_type_mapping[room_type]


    # Create a price prediction button
    if st.button('Predict Price'):
        # Call the function with the selected room_type as an argument
        rt_int = match_room_type(room_type)
                
        # Make the prediction   
        input_data = [[minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_int, n_int, rt_int, reviews_per_month, is_zero]]
        input_df = pd.DataFrame(input_data, columns=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count','availability_365','neighbourhood_group', 'neighbourhood', 'room_type', 'reviews_per_month', 'is_zero'])
        #prediction = rf.predict(input_df)   
        prediction = rf.predict(input_df)[0]
        
        # Populate the output dataframe with the input features and the predicted price
        #output_data = [minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_input, n_input, rt_int, reviews_per_month, is_zero, prediction[0]]

        # Create a dataframe with the input features and the predicted price
        #output_df = pd.DataFrame([output_data], columns=['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count','availability_365','neighbourhood_group', 'neighbourhood', 'room_type', 'reviews_per_month', 'is_zero', 'predicted_price'])


        # Make the prediction   
        # show prediction on price in dollars and cents using the price column 
        input_data = [[minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, ng_int, n_int, rt_int, reviews_per_month, is_zero]]

        # Make a prediction based on the user-defined input features
        # neighbourhood_group, neighbourhood, room_type, availability_365, calculated_host_listings_count, minimum_nights, number_of_reviews, reviews_per_month as features
        predicted_price = rf.predict(input_df)[0]
        
        # Format the price prediction
        predicted_price = '${:,.2f}'.format(predicted_price)
        
        # Display the price prediction
        #st.write('The predicted average price is ${:.2f}.'.format(predicted_price))
        st.write('The predicted average price is {}.'.format(predicted_price))
        #st.dataframe(output_df)
        
        #display(price_prediction_button)
