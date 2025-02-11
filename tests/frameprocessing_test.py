import cv2
import matplotlib.pyplot as plt
from scripts.preprocessing.FrameProcessor import FrameProcessor

img = cv2.imread('tests/assets/YOLODetector_test.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
FP = FrameProcessor()
img_yolo = FP.normalize_image_for_yolo(img)

plt.imshow(img)
plt.show()
plt.imshow(img_yolo)
plt.show()