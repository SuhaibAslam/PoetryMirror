from statistics import mode

import cv2 # https://pypi.org/project/opencv-python/
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

from timeit import default_timer as timer
import PySimpleGUI as sg

win_started = False

# time parameters
max_time = 3 # seconds

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting list for gathering modes over time per round
emotion_modes = []

# starting list for gathering mode of emotion modes over multiple rounds
emotion_modes_over_rounds = [] # RETURN THIS LIST IN THE END
rounds_so_far = 0

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(1)
start = timer()

while True:
    print("Round: ", rounds_so_far)
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
            emotion_modes.append(emotion_mode)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    imgbytes = cv2.imencode('.png', bgr_image)[1].tobytes()
    # ---------------------------- THE GUI ----------------------------
##    if not win_started:
##        win_started = True
##        layout = [
##            [sg.Text('Yolo Playback in PySimpleGUI Window', size=(30, 1))],
##            [sg.Image(data=imgbytes, key='_IMAGE_')],
##            [sg.Exit()]
##        ]
##        win = sg.Window('YOLO Webcam Demo', layout, default_element_size=(14, 1), text_justification='right', auto_size_text=False, finalize=True)
##        image_elem = win['_IMAGE_']
##    else:
##        image_elem.update(data=imgbytes)
##
##    event, values = win.read(timeout=0)
##    if event is None or event == 'Exit':
##        break
##    print("[INFO] cleaning up...")
##    win.close()
    # -----------------------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end = timer()
    time_elapsed = end - start
    if time_elapsed >= max_time:
        try:
            emotion_mode_this_round = mode(emotion_modes)
            emotion_modes_over_rounds.append(emotion_mode_this_round)
            rounds_so_far += 1
            emotion_modes = []
        except:
            start = timer()
            continue
    if rounds_so_far == 4:
        print("Emotions over four rounds:")
        print(emotion_modes_over_rounds)
        break


video_capture.release()
cv2.destroyAllWindows()


layout = 	[
		[sg.Text('YOLO')],
		[sg.Text('Confidence'), sg.Slider(range=(0,10),orientation='h', resolution=1, default_value=5, size=(15,15), key='confidence')],
		[sg.Text('Threshold'), sg.Slider(range=(0,10), orientation='h', resolution=1, default_value=3, size=(15,15), key='threshold')],
		[sg.OK(), sg.Cancel(), sg.Stretch()]
			]

win = sg.Window('YOLO',
				default_element_size=(14,1),
				text_justification='right',
				auto_size_text=False).Layout(layout)
event, values = win.Read()
args = values
print(args)
win.Close()

layout = [  [sg.Text('This is some poetry', font=('Traveling _Typewriter', 70))],
            [sg.Text('Enter something on Row 2'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel'), sg.Button('Exit')] ]


win = sg.Window('YOLO',
				default_element_size=(14,1),
				text_justification='right',
				auto_size_text=False).Layout(layout)
event, values = win.Read()
win.Close()

print(values)
