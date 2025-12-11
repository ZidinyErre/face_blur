import cv2 

img_path = 'maxresdefault.jpg'

img = cv2.imread(img_path)

img.shape
(4000, 2667, 3)


print(cv2.__version__)

