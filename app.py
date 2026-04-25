import streamlit as st
import pickle
import numpy as np

# 1. Load the "Brain" (Your saved model)
with open('gold_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Design the Website Interface
st.title("💰 Gold Price Prediction App")
st.write("Enter the market data below to predict the closing price.")

# 3. Create input boxes for the user
col1, col2 = st.columns(2)
with col1:
    open_price = st.number_input("Open Price")
    high_price = st.number_input("High Price")
with col2:
    low_price = st.number_input("Low Price")
    volume = st.number_input("Volume")

# 4. The Prediction Logic
if st.button("Predict Now"):
    # Arrange the inputs into the format the model expects
    features = np.array([[open_price, high_price, low_price, volume]])
    prediction = model.predict(features)
    
    # Show the result!
    st.success(f"The Predicted Closing Price is: **${prediction[0]:.2f}**")