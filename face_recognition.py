import dlib
import cv2
import numpy as np
import json

PREDICTOR_PATH = 'shape_predictor_5_face_landmarks.dat'
MODEL_PATH = 'dlib_face_recognition_resnet_model_v1.dat'
JSON_PATH = 'faces.json'

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)
model = dlib.face_recognition_model_v1(MODEL_PATH)
db_faces = dict()

def get_euclidean_distance(source: np.ndarray, new: np.ndarray):
    distance = np.sqrt(np.sum((source - new) ** 2))
    return distance

def load_faces(path: str):
    global db_faces
    with open(path, 'r') as f:
        db_faces = json.load(f)
    
    for name in db_faces:
        db_faces[name] = np.array(db_faces[name])   

def save_faces(path: str):
    global db_faces
    for name in db_faces:
        db_faces[name] = db_faces[name].tolist()
    with open(path, 'w') as f:
        json.dump(db_faces, f, indent=3)

def add_face(name: str, descriptor: np.ndarray):
    global db_faces
    if name and descriptor:
        db_faces[name] = descriptor

def remove_face(name: str):
    global db_faces
    if name in db_faces:
        db_faces.pop(name) 

def recognize(frame: np.ndarray):
    recognized_faces = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        label = 'Unknown'
        color = (75, 74, 224)
        shape = shape_predictor(gray, face) # 5 points
        aligned_face = dlib.get_face_chip(frame, shape)
        face_descriptor = model.compute_face_descriptor(aligned_face)
        face_descriptor = np.array(face_descriptor)
        distance = 1
        for name in db_faces:
            verified_face = db_faces[name]
            distance = get_euclidean_distance(verified_face, face_descriptor)
            if distance <= 0.5:
                label = name
                color = (229, 160, 21)
                break
        recognized_faces.append({'bbox': face, 'descriptor': face_descriptor,
        'distance': distance, 'color': color, 'name': label, 'shape': shape})

    return recognized_faces

def draw_prediction(frame: np.ndarray, recognized_faces: list, draw_shape: bool=False):
    for face in recognized_faces:
        x = face['bbox'].left() if face['bbox'].left() > 0 else 1
        y = face['bbox'].top() if face['bbox'].top() > 0 else 1
        w, h = face['bbox'].right() - face['bbox'].left(), face['bbox'].bottom() - face['bbox'].top()
        cv2.rectangle(frame, (x, y), (x + w, y + h), face['color'], 1)
        cv2.rectangle(frame, (x, y - 25), (x + w, y), face['color'], -1)
        
        if face['name'] != 'Unknown':
            label = face['name'] + ' ' + str(round(face['distance'], 2))
        else:
            label = face['name']
        cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        if draw_shape:
            for n in range(0, 5):
                x = face['shape'].part(n).x
                y = face['shape'].part(n).y
                cv2.circle(frame, (x, y), 3, (47, 235, 213), -1)

def main():
    
    load_faces(JSON_PATH)
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        recognized_faces = recognize(frame)
        draw_prediction(frame, recognized_faces, draw_shape=False)

        if len(recognized_faces):
            name = recognized_faces[-1]['name']
            face_descriptor = recognized_faces[-1]['descriptor']
        
        cv2.imshow('face recognition', frame)
        
        key = cv2.waitKey(1)
        if key == 32 and name == 'Unknown': # 'SPACE'
            name = str(input('Write new face\'s name: '))
            add_face(name, face_descriptor)
        elif key == 100: # 'd'
            remove_face(name)
            print(f'{name} was deleted')
        elif key == 27: # 'ESC'
            save_faces(JSON_PATH)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
