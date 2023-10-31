import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model with error handling
model_path = 'Desktop/SG_Research/Best_red_rot_sugarcane_model.h5'  # Update with the path to your model file
copyright_text = """
Copyright © 2023 [Saumyadeep Mitra](https://www.saumyadeepmitra.live). All rights reserved.
The app uses deep learning to classify sugarcane leaves as healthy or unhealthy based on the presence of Red Rot Disease.
"""
try:
    model = load_model(model_path)
except FileNotFoundError:
    st.error(f"Error: Model file '{model_path}' not found.")
    st.stop()

# Define a function for image classification
def classify_image(image_path, top_k=1, class_mapping=None):
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(128, 128))  # Resize the image to the expected input size
        img = img_to_array(img) / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict the class probabilities
        class_probs = model.predict(img)[0]

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

# Define a function to check if the leaf is healthy
def is_leaf_healthy(predicted_labels):
    return "Healthy" in predicted_labels[0]

# Streamlit app
st.set_page_config(
    page_title="Red Rot Sugarcane Disease Leaf Classifier 🌾",
    page_icon="🌾",
    layout="centered",
    menu_items={
        'Get Help': 'https://www.saumyadeepmitra.live',
        'Report a bug': 'https://www.saumyadeepmitra.live/contact',
        'About': st.markdown("""# Copyright © 2023 Saumyadeep Mitra. All rights reserved.
        
                The app uses deep learning to classify sugarcane
    leaves as healthy or unhealthy based on the
    presence of Red Rot Disease.)"""
    }
)

# Landing Page Description
st.markdown(
    """
    # 🌾 Red Rot Sugarcane Disease Leaf Classifier App

    ## 🔍 Purpose
    This app is designed to classify sugarcane leaves as either healthy or unhealthy based on the presence of Red Rot Disease, a common issue in sugarcane farming.

    ## 🚀 Technology Used
    - 🧠 Framework: TensorFlow and Keras for deep learning model development.
    - 🖼️ Model: A trained deep learning model is used for image classification.
    - 🌐 Web Interface: Created using Streamlit, a Python library for building web applications.

    ## 👨‍🌾 User-Friendly
    The app is user-friendly and doesn't require any technical expertise. Users can easily upload an image and receive classification results.

    ## 📋 Step-by-Step Guide
    The app provides users with a step-by-step guide on how to use it, making the process intuitive and straightforward.

    ## 📢 Feedback
    Users have the option to provide feedback or report any issues, allowing for continuous improvement of the app.

    ## 📄 Copyright
    """)
st.markdown(copyright_text)
st.markdown("""
    ## ℹ️ About
    Users can access an "About" section to learn more about the app's purpose and the technology behind it.

    The Red Rot Sugarcane Disease Leaf Classifier app combines the power of deep learning with a user-friendly interface to help farmers and enthusiasts identify unhealthy sugarcane leaves, ultimately contributing to healthier crops. 🌾🌱👨‍🌾
    """
)

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

# Feedback form
st.subheader("Feedback Form")
with st.form(key='feedback_form'):
    feedback_text = st.text_area("Please provide your feedback or report any issues:")
    submit_button = st.form_submit_button("Submit Feedback")

# Handle feedback submission (you can customize this part)
if submit_button:
    # Here, you can send the feedback to your preferred service or save it to a file
    st.success("Feedback submitted! Thank you for your input.")


# Step-by-Step Guide in Sidebar
st.sidebar.title("Step-by-Step Guide")
st.sidebar.markdown("Follow these steps to use the app:")
st.sidebar.markdown("1. Upload an image of a sugarcane leaf.")
st.sidebar.markdown("2. Wait for the app to process the image.")
st.sidebar.markdown("3. Review the classification results.")
st.sidebar.markdown("4. The app will tell you if the leaf is healthy or not.")
