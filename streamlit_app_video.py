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
   image = Image.open(uploaded_file)

   st.image(image, caption='Uploaded Image', use_column_width=True)    

   test_image = image.resize((160,160))
   test_image = preprocessing.image.img_to_array(test_image)
   test_image = np.expand_dims(test_image, axis=0)
   class_names = [
           'Healthy', 
           'Anomalous']
   Genrate_pred = st.button("Generate Prediction")    
   if Genrate_pred:
       predictions = model.predict(test_image)
       scores = tf.nn.softmax(predictions[0])
       if (0 < predictions < 5):
          st.title("Predicted Label for the image is Healthy")
       else:
          st.title("Predicted Label for the image is Anomalous")
        


uploaded_video = st.file_uploader("Choose video", type=["mp4", "avi"])
frame_skip = 300 # display every 300 frames

if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk
    cur_frame = 0
    success = True

    while success:
        success, frame = vidcap.read() # get next frame from video
        if cur_frame % frame_skip == 0: # only analyze every n=300 frames
            print('frame: {}'.format(cur_frame)) 
            pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
            st.image(pil_img)
          
            test_image = pil_img.resize((160,160))
            test_image = preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            class_names = [
                    'Healthy', 
                    'Anomalous']
            predictions = model.predict(test_image)
            scores = tf.nn.softmax(predictions[0])
            if (0 < predictions < 5):
                st.title("Predicted Label for the image is Healthy")
            else:
                st.title("Predicted Label for the image is Anomalous")
          
        cur_frame += 1
