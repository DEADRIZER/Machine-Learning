from cv2 import *

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# On lit test.jpg
img = imread("images_test\\test.jpg")

# changement vers un format non RGB
grayScaled_img = cvtColor(img, cv2.COLOR_RGB2GRAY)


# On récupère les coordonnées (x y w h) du visage sur la photo
face_coordinates = trained_face_data.detectMultiScale(grayScaled_img)
# print(face_coordinates)


# Pour chaque visages detecté on dessine un rectangle autour des visages
for (x, y, w, h) in face_coordinates:
    # param_1 = l'image sur laquel nous voulons afficher nos rectangles 
    # param_2 = un tuple des coordonnées en haut à gauche
    # param_3 = un tuple des coordonnées en bas à droit
    # param_4 = un tuple de couleur BGR
    # param_5 = L'épaisseur du rectangle
    rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    

# Affiche une image
imshow("test", img)     # affichage de l'image originale
# imshow("test", grayScaled_img)     # affichage de l'image grisé

# Attend la pression d'une touche
waitKey()
