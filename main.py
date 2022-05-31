import streamlit as st
from detect_for_st import *
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
import time

with open("data/coco.names", "r") as f:
    classes_list = [line.strip() for line in f.readlines()]

# Title
st.title("Object Detection use YoloR")
st.sidebar.title("Settings")

# Confidence
confidence = st.sidebar.slider("Confidence", min_value=0.1, max_value=1.0, value=0.4)
st.sidebar.markdown("________________________________________________________________")


# Custom classes
custom_classes = st.sidebar.checkbox("Use custom classes")

# Custom Classes
if custom_classes:
    allow_class = []
    class_list = st.sidebar.multiselect("Select custom classes", classes_list, default="car")
    for cl in class_list:
        allow_class.append(classes_list.index(cl))
else:
    allow_class = [2]    # car



# Choose type files
file = st.sidebar.radio("Choose type file", ["Image", "Video"])

if file == "Image":
    file_upload = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if file_upload is not None:
        image = Image.open(file_upload)
        img_array = np.array(image, dtype="uint8")
        st.image(img_array, use_column_width=True)


else:
    file_upload = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "m4v", "webm"])
    stframe = st.empty()
    tffile = tempfile.NamedTemporaryFile(delete=False)
    st.markdown("**Output**")
    if file_upload is not None:
        tffile.write(file_upload.read())
        cap = cv2.VideoCapture(tffile.name)
        pause = st.button("Pause")
    # Output
        kp1, kp2 = st.columns(2)
        with kp1:
            st.markdown("**Frame Rate**")
            kp1_text = st.markdown("0")

        with kp2:
            st.markdown("**Objects count**")
            kp2_text = st.markdown("0")

        while cap.isOpened():
            s = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            with torch.no_grad():
                frame = detect(frame, allow_class=allow_class, conf_thres=0.4)
            frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
            stframe.image(frame, channels="BGR", use_column_width=True)
            fps = round(1 / (time.time() - s))
            kp1_text.write(str(fps))
            while pause is False:


