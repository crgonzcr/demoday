import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import preprocessing
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/model.hdf5")
### load file
uploaded_file = st.file_uploader("Choose File", type=["png","jpg","jpeg"])

map_dict = {0: 'Healthy',
          1: 'Anomalous'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
   image = Image.open(uploaded_file)
          
#    img = tf.keras.preprocessing.image.load_img(image, target_size=(160,160))
#    img_array = tf.keras.preprocessing.image.img_to_array(img)
#    img_array = tf.expand_dims(img_array, 0)
#    pred = model.predict(img_array)

#     image = Image.open(uploaded_file)
   st.image(image, caption='Uploaded Image', use_column_width=True)    

   test_image = image.resize((160,160))
   test_image = preprocessing.image.img_to_array(test_image)
#    test_image = test_image / 127.5
   test_image = np.expand_dims(test_image, axis=0)
   class_names = [
           'Healthy', 
           'Anomalous']
          
#     file_bytes = np.asarray(image, dtype='uint8')
          
#     file_bytes = np.asarray(bytearray(image), dtype=np.uint8)
#     opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
#     resized = cv2.resize(opencv_image,(160,160))
#     #Now do something with the image! For example, let's display it:
#     st.image(opencv_image, channels="RGB")

#     resized = mobilenet_v2_preprocess_input(test_image)
#     img_reshape = resized[np.newaxis,...]

   Genrate_pred = st.button("Generate Prediction")    
   if Genrate_pred:
#        prediction = model.predict(resized)
       logits = model(test_image)
       predictions = model.predict(test_image)
       st.write(predictions)
       scores = tf.nn.softmax(predictions[0])
#        scores = scores.numpy()
       results = {
          'Healthy': 0,
          'Anomalous': 1
          }
       st.write(scores)
       result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence."
       st.title(result)

