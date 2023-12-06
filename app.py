from flask import Flask 
from flask import render_template 
from flask import request
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras import Sequential

import cv2
from keras.models import load_model
 
app = Flask(__name__)
UPLOAD_FOLDER = "C:\\Users\\riddh\\Desktop\\SIC\\static"

model = load_model("C:\\Users\\riddh\\Desktop\\SIC\\model1_vgg19.h5")

img_shape = [224,224]


def pred(path):

  img_array = cv2.imread(path)
  rgb_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
  resize_array = cv2.resize(rgb_array,img_shape)
  img_array = np.array(resize_array).reshape(-1,224,224)
  img_array = img_array / 255.00
  img_array = img_array.reshape(-1,224,224,3)

  predict = model.predict(img_array)
  label = np.argmax(predict,axis=1)
  prob = str(predict[0,label]*100) +'%'

  if label==0:
        ans = "Cloudy"
  elif label==1:
        ans= "Desert"
  elif label==2:
        ans= "Green Area"
  else:
        ans = "Water"

  return ans,prob

@app.route("/", methods = ["GET", "POST"])
def upload_pred():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            predict = pred(image_location)
            print(predict)
            return render_template("index.html", predict = predict, image_loc = image_file.filename)
    return render_template("index.html", predict = 0, image_loc = None)

if __name__ == "__main__":
    app.run(port = 12000, debug = True)
    

    