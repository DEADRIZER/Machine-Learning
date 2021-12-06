import cv2


trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# driver de la caméra par défaut de l'ordinateur
webCam = cv2.VideoCapture(0)


while True:
    frame_read, frame = webCam.read()
    # changement vers un format non RGB
    grayScaled_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # # On récupère les coordonnées (x y w h) du visage sur la photo
    face_coordinates = trained_face_data.detectMultiScale(grayScaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 8)
    # Affiche une image
    cv2.imshow("test", frame)     # affichage de l'image originale
    # Attend la pression d'une touche
    key = cv2.waitKey(1)
    if key == 27:
        break

