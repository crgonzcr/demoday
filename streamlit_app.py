import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import preprocessing
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/model.hdf5")

map_dict = {0: 'Healthy',
          1: 'Anomalous'}

st.title("Visual Anomaly Detection for Sewerage")

st.text("Upload a sewage video to classify it as healthy or anomalous")

uploaded_video = st.file_uploader("Choose video", type=["mp4", "avi"])
# frame_skip = 10 # display every 300 frames

if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
#     with open(vid, mode='wb') as f:
#         f.write(uploaded_video.read()) # save video to disk

#     st.markdown(f"""
#     ### Files
#     - {vid}
#     """,
#     unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk
    
    success = True

    st.text("Frames")
    frame = st.number_input("Selecciona el N° de frames:", )
    st.write(frame)
    cur_frame = 300
    frame_skip = 300
          
    Genrate_pred = st.button("Generate Prediction") 
    
    
    while success:
        success, frame = vidcap.read() # get next frame from video
        if cur_frame % frame_skip == 0: # only analyze every n=300 frames
            print('frame: {}'.format(cur_frame)) 
            if success:
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
            else:
                break
          
        cur_frame += 1
#         if success == False:
#             break
