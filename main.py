import cv2 
import matplotlib.pyplot as plt
from PIL import Image , ImageFilter
import numpy
# Importe OpenCV lecture d’images, vision par ordinateur.
# Importe Matplotlib → afficher l’image dans une fenêtre.

img_path = 'put your file her '
# chemin de l'image 
img = cv2.imread(img_path)
# Charge l’image en mémoire. Résultat = matrice de pixels (format BGR).
img.shape
(4000, 2667, 3)
# forme de la matrice :hauteur, largeur, 3 canaux couleur.

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Convertit l’image couleur en niveaux de gris.La détection de visage fonctionne mieux en gris.
gray_image.shape
(4000, 2667)
# plus qu’un seul canal (intensité).

face_classifier = cv2.CascadeClassifier( cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Charge un modèle pré-entraîné pour détecter des visages.

face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
# Cherche des visages dans l’image. Résultat :une liste de rectangles chaque rectangle = (x, y, w, h)

for (x, y , w, h) in face:
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    the_face = img[y: y+h, x: x+w]
    my_img = Image.fromarray(the_face)
    img_blur = my_img.filter(ImageFilter.GaussianBlur(12))
    numpy_img = numpy.array(img_blur)
    img[y: y+h, x: x+w] = numpy_img


# Il faut que je réinjecte dans l'img


# Boucle sur chaque visage détecté.


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convertit l’image BGR en RGB




plt.figure(figsize=(20, 10))
# Crée une fenêtre d’affichage grande.
plt.imshow(img_rgb)
# plt.imshow(numpy_img)
# Affiche l'image
plt.axis('off')
# enlève les axes (plus propre)
plt.show()

