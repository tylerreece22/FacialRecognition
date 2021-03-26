import numpy as np
import cv2
from PIL import Image

from res_facenet.models import model_920
from FaceVerification import FaceVerification

model920 = model_920()

haar = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

from gpiozero import LED
from gpiozero import MotionSensor

led = LED(17)
pir = MotionSensor(4)
led.off()


def face_detect(image):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 2, 2)

    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 10)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(image)

    return image


while True:
    pir.wait_for_motion()
    print('Motion detected')
    led.on()

    cap = cv2.VideoCapture(0)
    anchor_image = Image.open('./data/me.jpg')

    print('Ready!')
    frame_limit = 10
    frames = 0
    while frames < frame_limit:
        ret, frame = cap.read()
        if ret == False:
            print(f"Printed {frames} frames")
            break

        face_verification = FaceVerification(np.array(anchor_image), frame)
        print('Processed frame')

        if face_verification.verify_face():
            print('you may enter')
            break

        # cv2.imshow('object_detect',np.asarray(frame))
        if cv2.waitKey(40) == 27:  # Every 40 milliseconds check if escape key is pressed
            break
        frames += 1
        

    cv2.destroyAllWindows()
    cap.release()
    led.off()
