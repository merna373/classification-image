import streamlit as st
from PIL import Image
from transformers import pipeline

# Load the image classification pipeline
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="watersplash/waste-classification")

pipe = load_model()

# Set up the Streamlit application
st.title("Waste Classification App")
st.write("Upload an image of waste to classify it.")

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Classify the image
    with st.spinner("Classifying..."):
        results = pipe(image)

    # Display the classification results
    st.write("Classification Results:")
    for result in results:
        st.write(f"- {result['label']}: {result['score']:.2f}")