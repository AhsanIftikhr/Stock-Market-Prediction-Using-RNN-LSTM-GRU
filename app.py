import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load trained GRU model
model = load_model("gru_stock_prediction_model.h5")

# Page config
st.set_page_config(
    page_title="Stock Market Prediction",
    layout="centered"
)

# Title
st.title("📈 Stock Market Prediction App")

st.write("""
This application predicts the next stock price using a trained GRU Deep Learning model.
""")

st.subheader("Enter Last 60 Normalized Stock Prices")

# Text area input
input_text = st.text_area(
    "Enter 60 comma-separated values",
    height=150
)

# Predict button
if st.button("Predict Next Price"):

    try:

        values = list(map(float, input_text.split(',')))

        if len(values) != 60:

            st.error("Please enter exactly 60 values.")

        else:

            data = np.array(values)
            data = data.reshape(1, 60, 1)

            prediction = model.predict(data)

            st.success(
                f"Predicted Normalized Stock Price: {prediction[0][0]:.4f}"
            )

    except:

        st.error("Invalid input format.")