import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QFileDialog


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('cnn_vgg19_tl_final.h5')
emotion_dict = {
    0: 'Anger',
    1: 'Fear',
    2: 'Happy',
    3: 'Neutral',
    4: 'Sad',
    5: 'Surprise'
    }

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def convert_image(image):
    image_arr = []
    pic = cv2.resize(image, (256, 256))
    image_arr.append(pic)
    image_arr = np.array(image_arr)
    image_arr = image_arr.astype('float32')
    image_arr /= 255
    predict_x = model.predict(image_arr)
    classes_x = np.argmax(predict_x, axis=1)

    return classes_x


while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        gray = cv2.flip(frame, 1)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]

            prediction = int(convert_image(roi_gray))

            emotion = emotion_dict[prediction]

            cv2.putText(gray, emotion, (x + 10, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2  , cv2.LINE_AA)

        cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Video', gray)
        cv2.resizeWindow('Video', 1000, 600)

        if cv2.waitKey(1) == 27:  # press ESC to break
            cap.release()
            cv2.destroyAllWindows()
            break

    else:
        break