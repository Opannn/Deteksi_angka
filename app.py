import streamlit as st
from streamlit_drawable_canvas import st_canvas

from tensorflow import keras
from tensorflow.keras.models import load_model

import cv2
import numpy as np

# Load the pre-trained MNIST model
model = load_model('mnist.hdf5')  # Ensure this file exists in the same directory
st.write("Model loaded successfully!")
st.title("Deteksi Angka Tulis Tangan")

# Define canvas size
SIZE = 192

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="#ffffff",  # Brush color
    stroke_width=10,
    stroke_color='#ffffff',  # Brush stroke color
    background_color="#000000",  # Canvas background
    height=150,
    width=150,
    drawing_mode='freedraw',
    key="canvas",
)

# Process the image drawn on the canvas
if canvas_result.image_data is not None:
    # Resize and display the drawn image
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))  # Resize to MNIST size
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)  # Upscale for display
    st.write('Input Image')
    st.image(img_rescaling)

# Handle the Predict button click
if st.button('Predict'):
    # Ensure the image is processed before prediction
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    test_x = test_x / 255.0  # Normalize pixel values to [0, 1]
    test_x = test_x.reshape(1, 28, 28, 1).astype('float32')  # Reshape for the model

    # Debugging information
    st.write(f"Processed Input Shape: {test_x.shape}")
    st.write(f"Input Data Type: {test_x.dtype}")

    # Make the prediction
    pred = model.predict(test_x)
    result = np.argmax(pred[0])  # Get the predicted class
    st.write(f"Prediction Result: {result}")
    st.bar_chart(pred[0])  # Display prediction probabilities
