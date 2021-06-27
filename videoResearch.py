from cv2 import *


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# driver de la caméra par défaut de l'ordinateur
# webCam = VideoCapture(0)
webCam = VideoCapture("video_test\\test.mp4")


while True:
    frame_read, frame = webCam.read()

    # changement vers un format non RGB
    grayScaled_img = cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # # On récupère les coordonnées (x y w h) du visage sur la photo
    face_coordinates = trained_face_data.detectMultiScale(grayScaled_img)

    for (x, y, w, h) in face_coordinates:
        rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


    # Affiche une image
    imshow("test", frame)     # affichage de l'image originale

    # Attend la pression d'une touche
    key = waitKey(1)

    if key == 27:
        break