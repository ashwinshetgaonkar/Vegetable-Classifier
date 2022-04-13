import tensorflow as tf

from PIL import Image
import streamlit as st

model=tf.keras.models.load_model('best_model')


st.markdown("## Vegetable Classifier")
st.markdown("---")

class_names=['Bean','Bitter Gourd','Bottle Gourd','Brinjal','Broccoli','Cabbage','Capsicum','Carrot','Cauliflower','Cucumber','Papaya','Potato','Pumpkin','Radish','Tomato']
uploaded_file = st.file_uploader("Upload your Image you want to classify:", type=["png", "jpg", "jpeg"])
st.markdown(" ")
if st.button('Predict'):
    img = Image.open(uploaded_file)
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, size=[224, 224])
    img = tf.reshape(img, (1, 224, 224, 3))
    pred = tf.argmax(tf.squeeze(model.predict(img))).numpy()
    predicted_class=class_names[pred]
    st.markdown(" ")
    st.write("Prediction: It's a ", predicted_class)
else:
     pass





