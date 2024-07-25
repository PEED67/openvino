import streamlit as st
import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import ai
import io


def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    while(camera.isOpened()):
        ret, image = camera.read()
        if ret:
            image = np.array(image)
            if image.size == 0:
                continue
            if image is None:
                continue
            input_image = ai.preprocess(image, ai.input_layer_face)
            results = ai.compiled_model_face([input_image])[ai.output_layer_face]
            face_boxes, scores = ai.find_faceboxes(image, results, confidence_threshold)
            if len(face_boxes) > 0:
                output = ai.draw_ag_emo(face_boxes, image, source_radio)
                st_frame.image(output, channels = "BGR")
            else:
                st_frame.image(image, channels = "BGR")
        else:
            camera.release()
            break
    
st.set_page_config(
    page_title="Age/Gender/Emotions",
    page_icon=":sos:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("HELPPPPPPP :sos:")
st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE","VIDEO","WEBCAM"])

st.sidebar.header("Confidence")
confidence_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 5, 100, 20))/100

input = None
temporary_location = None
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg","png"))
    if input is not None:
        input_image = PIL.Image.open(input)
        image = np.array(input_image.copy())
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        input_image = ai.preprocess(image, ai.input_layer_face)
        results = ai.compiled_model_face([input_image])[ai.output_layer_face]
        face_boxes, scores = ai.find_faceboxes(image, results, confidence_threshold)
        output = ai.draw_ag_emo(face_boxes, image, source_radio)
        st.image(output)
    else:
        st.image("assets/sample_image.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")
        
if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an video.", type=("mp4"))
    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4"
        with open(temporary_location, "wb") as out:
            out.write(g.read())
        out.close()
    if temporary_location is not None:
        play_video(temporary_location)
        if st.button("Replay", type="primary"):
            pass
    else:
        st.video("assets/sample_video.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an video.")

if source_radio == "WEBCAM":
    play_video(0)