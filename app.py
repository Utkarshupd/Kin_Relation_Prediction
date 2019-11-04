# import the necessary packages

import numpy as np
import flask
import io
import seaborn as sns
import pandas as pd
from PIL import Image
import os
from random import choice, sample
import cv2
from imageio import imread
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras import regularizers
from keras.layers import Input, Embedding, LSTM, Dropout, BatchNormalization, Dense, concatenate, Flatten, Conv1D
from keras.optimizers import RMSprop, Adam

# from keras_vggface.vggface import VGGFace
from glob import glob
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing import image
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, \
    Subtract, Add, Conv2D, Lambda, Reshape
from collections import defaultdict
import tensorflow as tf
from sklearn.metrics import roc_auc_score
#from keras_vggface.utils import preprocess_input

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

model = load_model('facenet_vgg.h5', custom_objects={'auc': auc})
model._make_predict_function()
graph = tf.get_default_graph()


# Image preprocessing step
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y

#Image preprocessing step
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

#this function will read image from specified path and convert it into required size
def read_img_fn(path):
    # Facenet architecture will take image of size 160 x 160
    IMG_SIZE_FN = 160

    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_FN,IMG_SIZE_FN))
    img = np.array(img).astype(np.float)
    return prewhiten(img)

#this function will read image from specified path and convert it into required size
def read_img_vgg(path):
    # Facenet architecture will take image of size 224 x 224
    IMG_SIZE_VGG = 224

    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_VGG,IMG_SIZE_VGG))
    img = np.array(img).astype(np.float)
    return prewhiten(img)
    #return preprocess_input(img, version=2)

def prepare_image(image1, image2):

    X1_FN = np.array([read_img_fn(image1)])
    #X1_FN= np.expand_dims(X1_FN, axis=0)
    X1_VGG = np.array([read_img_vgg(image1)])
    #X1_VGG = np.expand_dims(X1_VGG, axis=0)

    X2_FN = np.array([read_img_fn(image2)])
    #X2_FN = np.expand_dims(X2_FN, axis=0)
    X2_VGG = np.array([read_img_vgg(image2)])
    #X2_VGG = np.expand_dims(X2_VGG, axis=0)

    # return the processed images
    return [X1_FN, X2_FN, X1_VGG, X2_VGG]


@app.route('/')
@app.route('/home')
def upload_image():
    return flask.render_template('index.html')


# example of face detection with mtcnn
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


@app.route('/upload', methods=['POST']) #POST will get the data and perform operatins
def post_image():
    global graph

    if flask.request.method == 'POST':

        target=os.path.join(APP_ROOT, 'static/')
        print(target)
        print(flask.request.files.getlist('file1'))
        for upload in flask.request.files.getlist("file1"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename1 = upload.filename
            # This is to verify files are supported
            ext = os.path.splitext(filename1)[1]
            if (ext == ".jpg") or (ext == ".png"):
                print("File supported moving on...")
            else:
                return 'Image Not Uploaded'
            destination = "/".join([target, filename1])
            print("Accept incoming file:", filename1)
            print("Save it to:", destination)
            upload.save(destination)


        target = os.path.join(APP_ROOT, 'static/')
        print(target)
        print(flask.request.files.getlist('file2'))
        for upload in flask.request.files.getlist("file2"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename2 = upload.filename
            # This is to verify files are supported
            ext = os.path.splitext(filename2)[1]
            if (ext == ".jpg") or (ext == ".png"):
                print("File supported moving on...")
            else:
                return 'Image Not Uploaded'
            destination = "/".join([target, filename2])
            print("Accept incoming file:", filename2)
            print("Save it to:", destination)
            upload.save(destination)
            filename=[filename1,filename2]
            target = os.path.join(APP_ROOT, 'static/')
            dirs = os.listdir(target)
            image1 = os.path.join(target, dirs[0])
            image2 = os.path.join(target, dirs[1])
            print(dirs)

            X1_FN, X2_FN, X1_VGG, X2_VGG = prepare_image(image1, image2)

            with graph.as_default():
                outputs = model.predict([X1_FN, X2_FN, X1_VGG, X2_VGG])

            return flask.render_template('end.html',image_name=filename, pred=outputs)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    app.run(debug=True)
