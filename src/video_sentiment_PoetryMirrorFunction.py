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

# time parameters
max_time = 7 # seconds

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
# print(emotion_labels) # {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
# emotion_labeles = {0: 'negative', 1: 'negative', 2: 'negative', 3: 'positive', 4: 'negative', 5: 'positive', 6: 'neutral'}
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
emotion_mode_over_rounds = "neutral"

# starting video streaming
def facial_sentiment_runner():

    cv2.namedWindow('window_frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window_frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    video_capture = cv2.VideoCapture(0)
    start = timer()
    
    while True:
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end = timer()
        time_elapsed = end - start
        # print(time_elapsed)
        if time_elapsed >= max_time:
            try:
                emotion_mode_over_rounds = mode(emotion_modes)
                break
            except:
                start = timer()
                continue
    print("Emotion chosen:")
    print(emotion_mode_over_rounds)
    
    valence_dict = {'angry': 'negative', 'disgust':'negative', 'fear':'negative', 'happy':'positive', 'sad':'negative', 'surprise':'positive', 'neutral':'neutral'}
    overall_valence_for_poem = valence_dict[emotion_mode_over_rounds]
    
    print("Valence chosen:")
    print(overall_valence_for_poem)
    video_capture.release()
    cv2.destroyAllWindows()
    return overall_valence_for_poem    
#
#facial_sentiment_runner()