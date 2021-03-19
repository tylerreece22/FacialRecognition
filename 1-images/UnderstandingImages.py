import numpy as np # numerical library
from PIL import Image # python image library
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Reading image using PIL
#image = Image.open('../test.jpg')

image = cv2.imread('../test.jpg')

print('Image shape ', image.shape)

plt.imshow(image)
