import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.functional as F
from torchvision import transforms

from res_facenet.models import model_920

model920 = model_920()

haar = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')


class FaceVerification:
    def __init__(self, anchor_image, verification_image):
        # Should both be np.arrays
        self.anchor_image = anchor_image
        self.verification_image = verification_image

    @staticmethod
    def __pil_to_cv2(pil):
        return np.array(pil)

    @staticmethod
    def __cv2_to_pil(cv2_image):
        return Image.fromarray(cv2_image)

    # Returns pil image of detected face in verification image
    def __face_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, 2, 2)

        for x, y, w, h in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 10)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.__cv2_to_pil(image)

    def verify_face(self):
        # prepare preprocess pipeline
        preprocess_pipelines = [transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]
        transform = transforms.Compose(preprocess_pipelines)
        transforms.ToPILImage()
        transforms.Compose(preprocess_pipelines[:-1])

        # read the image and transform it into tensor then normalize it with our transform function pipeline
        anchor_image = transform(self.__face_detect(self.anchor_image)).unsqueeze(0)
        verification_image = transform(self.__face_detect(self.verification_image)).unsqueeze(0)

        # do forward pass
        anchor_image_embed, verification_image_embed = model920(anchor_image), model920(verification_image)

        # compute the distance using euclidean distance of image embeddings
        euclidean_distance = F.pairwise_distance(anchor_image_embed, verification_image_embed)

        # we use 1.5 threshold to decide whether images are genuine or impostor

        threshold = 1.5

        genuine = euclidean_distance <= threshold

        return genuine.item()

    def __get_face_from_webcam(self):
        cap = cv2.VideoCapture(0)
        anchor_image = Image.open('./data/me.jpg')

        frame_images = []
        frame_limit = 5
        frames = 0
        while frames < frame_limit:
            ret, frame = cap.read()
            if not ret:
                print(f"Printed {frames} frames")
                break

            frame_images.append(frame)
            # cv2.imshow('object_detect',np.asarray(frame))
            if cv2.waitKey(40) == 27:  # Every 40 milliseconds check if escape key is pressed
                break
            frames += 1

        cv2.destroyAllWindows()
        cap.release()
        return frame_images
