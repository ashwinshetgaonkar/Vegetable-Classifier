import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import mixed_precision
from PIL import Image
import streamlit as st



data_augmentation = keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2),
    tf.keras.layers.RandomRotation(0.2, fill_mode='nearest'),
    tf.keras.layers.RandomZoom(0.2),
    # layers.Rescaling(scale=1.0/255)
], name='Data_Augmentation_Layer')

tf.keras.mixed_precision.set_global_policy('mixed_float16')

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer')

base_model = keras.applications.efficientnet.EfficientNetB0(include_top=False)
base_model.trainable = False

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D(name='Global_Average_Pool_2D')(x)
num_classes = 15
outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype=tf.float32, name="Output_layer")(x)

model = keras.Model(inputs, outputs, name="model")
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)


model.load_weights('ModelCheckPoints/model_1.ckpt')


model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

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
    st.write("Prediction:It's a ", predicted_class)
else:
     pass





