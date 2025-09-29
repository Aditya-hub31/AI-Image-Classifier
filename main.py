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

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"] )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)


        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image"):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.success("Top Predictions:")
                    for i, (_, label, score) in enumerate(predictions):
                        st.markdown(f"**{i+1}. {label}** — `{score:.2%}`")

        if st.button("Clear"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
    




