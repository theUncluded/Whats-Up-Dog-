import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import fastai.vision.all 
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer


# Load your trained model
#path = path('dogdata')
#learn = load_learner(path / 'export.pkl')
# learn = tf.keras.models.load_model('./dogdata')
# learn2 = tf.keras.models.load_model('./dog-emotions-prediction')

st.title("Emotion Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((384, 384))  # adjust size as needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = learn2.predict(img_array)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Display the prediction
    emotion_labels = ['Relaxed', 'Happy', 'Sad', 'Anger']
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_emotion}")