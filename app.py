import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.image as mpimg

# Load the trained model with error handling
model_path = 'Desktop/SG_Research/Best_red_rot_sugarcane_model.h5'  # Update with the path to your model file
try:
    model = load_model(model_path)
except FileNotFoundError:
    st.error(f"Error: Model file '{model_path}' not found.")
    st.stop()

# Define a function for image classification
def classify_image(image_path, top_k=1, class_mapping=None):
    # Load and preprocess the image
    try:
        img = load_img(image_path, target_size=(128, 128))  # Resize the image to the expected input size
        img = img_to_array(img) / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Add this code before model prediction for debugging
        st.write(f'Image Shape: {img.shape}')

        # Predict the class probabilities
        class_probs = model.predict(img)[0]

        # Add this code after model prediction for debugging
        st.write(f'Class Probabilities: {class_probs}')

        # Get the top-k predicted class labels and probabilities
        top_indices = class_probs.argsort()[-top_k:][::-1]
        top_labels = [str(idx) for idx in top_indices]
        top_probabilities = [class_probs[idx] for idx in top_indices]

        if class_mapping:
            top_labels = [class_mapping.get(label, label) for label in top_labels]

        return top_labels, top_probabilities
    except FileNotFoundError:
        st.error(f"Error: Image file '{image_path}' not found.")
        return

# Rest of your code remains the same

# Define a function to check if the leaf is healthy
def is_leaf_healthy(predicted_labels):
    # Check if the top predicted label is "Healthy"
    return "Healthy" in predicted_labels[0]

# Streamlit app
st.title("Red Rot Sugarcane Disease Classifier")

# Upload an image for classification
uploaded_image = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.markdown("**Uploaded Image**")
    st.image(uploaded_image, use_column_width=True)

    # Check if the leaf is healthy and display the results
    top_labels, top_probabilities = classify_image(uploaded_image, top_k=3, class_mapping={0: 'Healthy', 1: 'Unhealthy'})
    leaf_is_healthy = is_leaf_healthy(top_labels)

    st.markdown("**Classification Results**")
    st.subheader("Top Predicted Classes and Probabilities:")
    for label, prob in zip(top_labels, top_probabilities):
        st.write(f'Class: {label}, Probability: {prob:.2f}')

    if leaf_is_healthy:
        st.success('The leaf is healthy.')
    else:
        st.error('The leaf is not healthy.')
