import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image 

def load_model():
    model = MobileNetV2(weights= "imagenet")
    return model

@st.cache_data 
def preprocess_image(image):   # converting image to what MobileNetV2 expects
    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)  # convert to numpy array
    img = cv2.resize(img, (224, 224))  # resize to 224x224
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0] #so this decode  line take the numeric predictions of mobilenetv2 and convert them to human readable format/ called as string labels then top = 3 means it takes top 3 predictions and indexing wil be start at 0 [0] means first prediction
        return decoded_predictions
    except Exception as e:
        st.error(f"Error in Classifiying image: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon = " 🖼️", layout="centered")
    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what's in it!")


    @st.cache_resource
    def load_cached_model():
        return  load_model()
     

    model = load_cached_model()

    uploaded_files = st.file_uploader(
    "Choose one or more images",
    type=["jpg", "png"],
    accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            
            if st.button(f"Classify {uploaded_file.name}"):
                with st.spinner("Analyzing Image..."):
                    predictions = classify_image(model, image)

                    if predictions:
                        st.success(f"Top Predictions for {uploaded_file.name}:")
                        for i, (_, label, score) in enumerate(predictions):
                            st.markdown(f"**{i+1}. {label}** — `{score:.2%}`")

        if st.button("Clear All"):
            st.experimental_rerun()

    st.markdown("---")  # separator line
    st.subheader("Or capture an image using your camera")

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)

        if st.button("Classify Camera Image"):
            with st.spinner("Analyzing Camera Image..."):
                predictions = classify_image(model, image)

                if predictions:
                    st.success("Top Predictions for Captured Image:")
                    for i, (_, label, score) in enumerate(predictions):
                        st.markdown(f"**{i+1}. {label}** — `{score:.2%}`")        

if __name__ == "__main__":
    main()
    




