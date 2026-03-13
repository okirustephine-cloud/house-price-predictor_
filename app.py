
import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import joblib
# Page configuration
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model


@st.cache_resource
def load_model():
    model = joblib.load("california_pipeline.pkl")
    return model

# Main app


def main():
    # Header
    st.markdown('<p class="main-header"> California Housing Price Predictor</p>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Welcome 
    This app predicts median house values in California based on census data from 1990.
    Simply adjust the parameters in the sidebar and get instant predictions!
    """)

    # Load model
    try:
        model = load_model()
        model_loaded = True
    except FileNotFoundError:
        st.error(
            " Model file not found! Please ensure 'california_knn_pipeline.pkl' is in the same directory.")
        model_loaded = False
        return

    # Sidebar inputs
    st.sidebar.header(" Input Features")
    st.sidebar.markdown("Adjust the values to predict house prices:")

    with st.sidebar:
        # Income
        med_inc = st.slider(
            "Median Income ($10k scale)",
            min_value=0.5, max_value=15.0, value=3.0, step=0.1,
            help="Median income in the block group"
        )

        # House Age
        house_age = st.slider(
            "House Age (years)",
            min_value=1, max_value=52, value=20, step=1,
            help="Median house age in the block group"
        )

        # Rooms
        ave_rooms = st.slider(
            "Average Rooms",
            min_value=2.0, max_value=10.0, value=5.0, step=0.1,
            help="Average number of rooms per household"
        )

        # Bedrooms
        ave_bedrms = st.slider(
            "Average Bedrooms",
            min_value=0.5, max_value=5.0, value=1.0, step=0.1,
            help="Average number of bedrooms per household"
        )

        # Population
        population = st.slider(
            "Population",
            min_value=100, max_value=35000, value=1000, step=100,
            help="Block group population"
        )

        # Occupancy
        ave_occup = st.slider(
            "Average Occupancy",
            min_value=1.0, max_value=10.0, value=3.0, step=0.1,
            help="Average number of household members"
        )

        # Location
        st.markdown("---")
        st.subheader(" Location")

        latitude = st.slider(
            "Latitude",
            min_value=32.0, max_value=42.0, value=37.0, step=0.01,
            help="Block group latitude"
        )

        longitude = st.slider(
            "Longitude",
            min_value=-124.5, max_value=-114.0, value=-122.0, step=0.01,
            help="Block group longitude"
        )

    # Create input DataFrame
    input_data = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [ave_rooms],
        'AveBedrms': [ave_bedrms],
        'Population': [population],
        'AveOccup': [ave_occup],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(" Input Summary")
        st.dataframe(input_data.style.format(
            "{:.2f}"), use_container_width=True)

        # Feature importance note
        st.info("""
         **Tip:** Median Income (MedInc) is typically the most influential 
        feature for house price predictions in this dataset.
        """)

    with col2:
        st.subheader(" Prediction")

        if st.button(" Predict Price", type="primary", use_container_width=True):
            # Make prediction
            prediction = model.predict(input_data)[0]
            price_usd = prediction * 100000

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin:0; color:#1f77b4;">Predicted Median House Value</h3>
                <h1 style="font-size: 3rem; margin: 10px 0; color: #2c3e50;">
                    ${price_usd:,.0f}
                </h1>
                <p style="font-size: 1.2rem; color: #7f8c8d;">
                    ({prediction:.3f} in $100,000s)
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Price interpretation
            if prediction < 1.5:
                st.error(" **Low Value Area** - Below average pricing")
            elif prediction < 3.0:
                st.warning("7**Moderate Value Area** - Average pricing")
            elif prediction < 4.5:
                st.success(" **High Value Area** - Above average pricing")
            else:
                st.balloons()
                st.success("🌟 **Premium Area** - Very high value!")

    # Model information
    st.markdown("---")
    st.subheader(" Model Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Algorithm</h4>
            <p>K-Nearest Neighbors Regressor</p>
        </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Best Parameters</h4>
            <p>• n_neighbors: 5<br>• weights: distance<br>• metric: Euclidean</p>
        </div>
        """, unsafe_allow_html=True)

    with info_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Performance</h4>
            <p>• R² Score: ~0.70<br>• RMSE: ~$75,000</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #7f8c8d;">
        Built with love and pasion| California Housing Dataset (1990 Census)
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

