import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
core = ov.Core()
model_face = core.read_model(model='models/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model = model_face, device_name="CPU")
input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

model_emo = core.read_model(model='models/emotions-recognition-retail-0003.xml')
compiled_model_emo = core.compile_model(model = model_emo, device_name="GPU")
input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

model_ag = core.read_model(model='models/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model = model_ag, device_name="GPU")
input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output(0)

def draw_ag_emo(face_boxes, image, type):
    show_image = image.copy()
    fontScale = show_image.shape[1]/750 
    EMOTION_NAMES = ['nomal', 'happy', 'sad', 'suprise', 'anger']
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        if xmin < 0 or ymin < 0 or xmax > image.shape[1] or ymax > image.shape[0]:
            return show_image
            continue
        face = image[ymin:ymax, xmin:xmax]
        input_image_emo = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image_emo])[output_layer_emo]
        results_emo = results_emo.squeeze()
        index = np.argmax(results_emo)
        input_image_ag = preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        age, gender = results_ag[1], results_ag[0]
        age = np.squeeze(age)
        age = int(age*100)
        gender = np.squeeze(gender)
        if (gender[0]>=0.78):
            gender = "female"
            if type == "IMAGE":
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
        elif (gender[1]>=0.68):
            gender = "male"
            if type == "IMAGE":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
        else:
            gender = "unknown"
            color = (0, 0, 0)
        text = gender + ' ' + str(age) + ' ' + EMOTION_NAMES[index]
        cv2.rectangle(show_image, (xmin,ymin), (xmax,ymax), color, int(fontScale*3))
        cv2.putText(show_image, text, (xmin, ymin-30), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, int(fontScale*3))
    return show_image
def preprocess(image, input_layer_face):
    N, input_channels, input_height, input_width = input_layer_face.shape
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    return input_image
def find_faceboxes(image, results, confidence_threshold):
    results = results.squeeze()
    scores = results[:,2]
    boxes = results[:,-4:]
    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    image_h, image_w, image_channels = image.shape
    face_boxes = face_boxes*np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)
    return face_boxes, scores