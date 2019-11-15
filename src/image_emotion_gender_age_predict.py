import cv2
import numpy as np
from keras.models import load_model
from time import time
import os
from mtcnn.mtcnn import MTCNN
from keras.preprocessing.image import img_to_array


emotion_labels_fer={0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'}
gender_labels = {0:'female', 1:'male'}

def get_img_path_list(dirs):
    img_path_list = []
    if not os.path.exists(dirs):
        return None
    for root, dirs, files in os.walk(dirs):
        for img in files:
            img_path = os.path.join(root, img)
            img_path_list.append(img_path)

    return img_path_list


def file_is_img(img_path):
    endswith = ['jpg', 'png', 'jpeg', 'gif', 'bmp']
    suffix = img_path.split('.') [-1].lower()
    if suffix in endswith:
        return True
    else:
        return False

class EmotionGenderAgeModel(object):
    """
    select the emotion classfier model had been trained
    """
    def __init__(self, cv2_model_path, emotion_model_path, gender_age_model_path):
        face_cascade = cv2.CascadeClassifier(cv2_model_path)
        self.face_cascade = face_cascade
        self.detector = MTCNN()
        emotion_classifier = load_model(emotion_model_path)
        self.emotion_classifier = emotion_classifier
        geder_age_classifier = load_model(gender_age_model_path)
        self.geder_age_classifier = geder_age_classifier
        
    def face_detect_by_CascadeClassifier(self, img):
        face_boxes = []
        if img.shape[2] != 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None

        for index, face in enumerate(faces):
            (x, y, w, h) = face
            if w <24 or h < 24:
                continue
            face_boxes.append([x, y, w, h])
        return face_box

    def face_detect_by_mtcnn(self, img):
        face_boxes = []
        #face detect and alignment
        faces = self.detector.detect_faces(img)
        if len(faces) == 0:
            return None
        
        for index, face in enumerate(faces):
            box = face["box"]
            confidence = face["confidence"]
            keypoints = face["keypoints"]
            if confidence < 0.6:
                continue
            x,y,w,h = box
            if w <24 or h < 24:
                continue
            face_boxes.append(box)
        return face_boxes
        
    def detect_faces(self, img_path, offset = 20):
        faces = []
        rects = []
        if not os.path.exists(img_path):
            return None
        if file_is_img(img_path):
            img = cv2.imread(img_path)
            if img is None:
                return None, None
            #优先采用mtcnn检测人脸，若是漏检则采用级联分类器检测一次
            face_boxes = self.face_detect_by_mtcnn(img)
            if face_boxes == None:
               face_boxes = self.face_detect_by_mtcnn(img)
            if face_boxes == None or len(face_boxes) == 0:
                return None, None

            for index, face in enumerate(face_boxes):
                x, y, w, h = face
                face_box = img[max(y-offset, 0): min(y+h+offset,img.shape[0]) , max(x-offset, 0):min(x+w+offset,img.shape[1])]
                faces.append(face_box)
                #x,y,w,h
                x = max(x-offset, 0)
                y = max(y-offset, 0)
                if w+x+offset >img.shape[1]:
                    w = img.shape[1] - x
                else:
                    w = w+offset
                if y+h+offset >img.shape[0]:
                    h = img.shape[0] - x
                else:
                    h = h+offset
                rects.append([x, y, w, h])
            return faces, rects
        
    def gender_age_model_predict(self, img, face_size =(64, 64)):
        img = cv2.resize(img, face_size)
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        (gender_prediction, age_prediction) = self.geder_age_classifier.predict(img)
        gender_label_arg = np.argmax(gender_prediction[0])
        gender_label = gender_labels[gender_label_arg]
        
        age_label_arg = np.argmax(age_prediction[0])
        age_label = str(age_label_arg)

        return gender_label, age_label
        
    def emotion_model_predict(self, img, face_size =(64, 64)):
        img = cv2.resize(img, face_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        emotion_prediction = self.emotion_classifier.predict(img)
        emotion_label_arg = np.argmax(emotion_prediction[0])
        emotion_label = emotion_labels_fer[emotion_label_arg]
        return emotion_label

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def draw_text(image_array, text, color, x, y,font_scale=2, thickness=2):
    cv2.putText(image_array, text, (x, y),cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

if __name__ == '__main__':
    color = (0, 255, 0)
    # starting video streaming
    image_dir = '../test_image/'
    cv2_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    gender_age_model_path = '../trained_models/gender_age_models/imdb_gender_age_XCEPTION_09-0.90-0.06.h5'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION_32-0.62.h5'
    
    emotion_gender_age_model = EmotionGenderAgeModel(cv2_model_path, emotion_model_path, gender_age_model_path)
    for image_path in get_img_path_list(image_dir):
        faces, rects = emotion_gender_age_model.detect_faces(image_path)
        if faces != None and len(faces)>0:
            for index in range(len(faces)):
                face = faces[index]
                rect = rects[index]
                #cv2.imshow('test', face)
                gender_label, age_label = emotion_gender_age_model.gender_age_model_predict(face, face_size =(64, 64))
                emotion_label = emotion_gender_age_model.emotion_model_predict(face, face_size =(64, 64))
                bgr_image = cv2.imread(image_path)
                draw_bounding_box(rect, bgr_image, color)
                draw_text(bgr_image, emotion_label, color, rect[0], rect[1]-50,font_scale=1, thickness=1)
                draw_text(bgr_image, gender_label, color, rect[0], rect[1]-30,font_scale=1, thickness=1)
                draw_text(bgr_image, age_label, color, rect[0], rect[1]-10,font_scale=1, thickness=1)
            cv2.imshow('window_frame', bgr_image)
            cv2.imwrite(image_path.replace('.jpg', '_result.jpg'), bgr_image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
