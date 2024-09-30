import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Set the page configuration with custom title and icon
st.set_page_config(page_title='🚗 CarDekho Price Prediction', page_icon='🚗', layout='wide')

# Custom CSS for enhanced UI styling
st.markdown("""
    <style>
    /* App title customization */
    .title-text {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ffa07a;
        text-shadow: 2px 2px 4px #000000;
    }
            
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1e90ff; /* blue for sidebar */
        padding: 20px;
    }

    /* Box shadow for cards */
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        margin-top: 20px;
    }

    /* Predict button styling */
    .stButton>button {
        background-color: #3498DB;
        color: white;
        padding: 12px 24px;
        margin-top: 20px;
        border-radius: 8px;
        border: none;
        font-size: 18px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980B9;
        transform: scale(1.05);
        color: #ffffff;
    }

    /* Sidebar font customization */
    .sidebar-text {
        font-size: 1.1rem;
        color: #2C3E50;
        font-weight: bold;
    }

    /* Loading spinner for predictions */
    .stSpinner {
        border: 16px solid #f3f3f3;
        border-top: 16px solid #3498DB;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Hover effects for inputs */
    input:hover {
        border-color: #3498DB;
    }

    </style>
    """, unsafe_allow_html=True)

# App title in bold and center alignment
st.markdown("<div class='title-text'>🚗 Car Price Prediction App</div>", unsafe_allow_html=True)

# Load model and encoders
with open(r"F:\GUVI\Project\models\Car_Dekho models\random_forest_model.pkl", 'rb') as m:
    rfr = pickle.load(m)

with open(r"F:\GUVI\Project\models\Car_Dekho models\minmax_scaler_features.pkl", 'rb') as f:
    mm_features = pickle.load(f)

with open(r"F:\GUVI\Project\models\Car_Dekho models\minmax_scaler_price.pkl", 'rb') as p:
    mm_price = pickle.load(p)

with open(r"F:\GUVI\Project\models\Car_Dekho models\label_encoders.pkl", 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Layout using 2 columns for better organization
col1, col2 = st.columns(2)


with st.sidebar:
    st.header('Choose Car Specifications')

    # Dropdowns and inputs in the sidebar
    city = st.selectbox('City', ['delhi','hyderabad','bangalore','chennai','kolkata','jaipur'])
    body_type = st.selectbox('Body Type', ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Minivans', 'Coupe', 'Hybrids'])
    transmission_type = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Electric', 'LPG'])
    insurance_validity = st.selectbox('Insurance Validity', ['Comprehensive', 'Third Party insurance', 'First Party insurance'])
    steering_type = st.selectbox('Steering Type', ['Power', 'Manual', 'Electric'])
    
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.header("Performance & Features")
    kilometers_driven = st.number_input('📏 Kilometers Driven', min_value=0, max_value=1000000, step=1000, value=15000)
    number_of_owners = st.number_input('👥 Number of Owners', min_value=1, max_value=10, step=1, value=1)
    year_of_manufacture = st.number_input('📅 Year of Manufacture', min_value=2000, max_value=2024, step=1, value=2016)
    seats = st.number_input('🪑 Seats', min_value=2, max_value=10, step=1, value=5)
    engine_size = st.number_input('⚙️ Engine Size (CC)', min_value=800, max_value=5000, step=100, value=1300)
    mileage = st.number_input('🛣 Mileage (km/l)', min_value=5.0, max_value=50.0, step=0.1, value=15.5)
    top_speed = st.number_input('🏎️ Top Speed (km/h)', min_value=50.0, max_value=350.0, step=10.0, value=150.0)
    cargo_volume = st.number_input('📦 Cargo Volume (Liters)', min_value=100, max_value=2000, step=50, value=350)

    st.markdown("</div>", unsafe_allow_html=True)

# Create feature input data for prediction
numerical_features = pd.DataFrame({
    'Kilometers driven': [kilometers_driven],
    'Number of Owners': [number_of_owners],
    'Year of manufacture': [year_of_manufacture],
    'Seats': [seats],
    'Engine': [engine_size],
    'Mileage': [mileage],
    'Cargo Volumn': [cargo_volume]
})

# Scale the numerical features
numerical_features_scaled = mm_features.transform(numerical_features)

top_speed_df = pd.DataFrame({
    'Top Speed': [top_speed]
})

# Prepare categorical input data
categorical_data = pd.DataFrame({
    'Body type': [body_type],
    'Transmission type': [transmission_type],
    'Fuel Type': [fuel_type],
    'Insurance Validity': [insurance_validity],
    'Steering Type': [steering_type],
    'City': [city]
})

# Transform categorical features using LabelEncoders
for column in ['Body type', 'Transmission type', 'Fuel Type', 'Insurance Validity', 'Steering Type', 'City']:
    categorical_data[column] = label_encoders[column].transform(categorical_data[column])

# Combine the scaled numerical features and encoded categorical features
input_data = pd.concat([pd.DataFrame(numerical_features_scaled, columns=numerical_features.columns),
                         top_speed_df.reset_index(drop=True),
                         categorical_data.reset_index(drop=True)], axis=1)

expected_feature_names = ["Body type",'Kilometers driven','Transmission type', 'Number of Owners', 'Year of manufacture',
                          'Fuel Type','Insurance Validity','Seats','Engine','Mileage','Steering Type','Top Speed','Cargo Volumn',
                          'City'
                           ]  # Prediction columns are order like when the model is trained

input_data = input_data[expected_feature_names]


with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Predict button with spinner/loading animation
    if st.button('🔍Calculate Price'):
        with st.spinner('💡 Predicting price...'):
            predicted_scaled = rfr.predict(input_data)
            predicted_price = mm_price.inverse_transform(predicted_scaled.reshape(-1, 1))

            st.success(f"🚗 The price is: ₹ {predicted_price[0][0]:,.2f} lakh")
    st.markdown("</div>", unsafe_allow_html=True)