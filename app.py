
from pywebio.platform.flask import webio_view
from flask import Flask
from pywebio.input import file_upload
from pywebio.output import put_image,put_tabs
from pywebio import start_server
import tensorflow as tf
import argparse
model = tf.keras.models.load_model('best_model')

app = Flask(__name__)


def predict_class():
    class_names = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot',
                   'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    image = file_upload("Upload an Image to Classify:", accept="image/*")
    img = tf.io.decode_image(image['content'])
    img = tf.image.resize(img, size=[224, 224])
    img = tf.reshape(img, (1, 224, 224, 3))

    pred = tf.argmax(tf.squeeze(model.predict(img))).numpy()
    predicted_class = class_names[pred]

    text = 'Predicted Class : ' + predicted_class
    put_tabs([
        {'title': 'Result', 'content': text},
        {'title': 'Uploaded Image', 'content': [
            put_image(image['content'], width='300px', height='300px', title='Vegetable')
        ]},
    ])


app.add_url_rule('/tool', 'webio_view', webio_view(predict_class),
                 methods=['GET', 'POST', 'OPTIONS'])

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-p","--port",type=int,default=8080)
    args=parser.parse_args()
    start_server(predict_class,port=args.port)



