import cv2



def save_webcam(outPath, fps, mirror=False):
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)
    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outPath, fourcc, fps, (int(width), int(height)))

    # initialize the first frame in the video stream
    firstFrame = None

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if mirror == True:
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)
            # Saves for video
            out.write(frame)
            # Display the resulting frame
            cv2.imshow('1. Frame simple', frame)
        else:
            break

        # Transformer l'image en noir et blanc
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("2. Frame grise",gray_frame)
        # Réduction du bruit de l'image en la lissant
        gray = cv2.GaussianBlur(gray_frame, (15,15), 0)
        cv2.imshow("3. Frame grise lissee",gray)

        # Initialisation de firstFrame au premier tour
        if firstFrame is None:
            firstFrame = gray
            continue

        # Faire la difference entre l'image de base et l'image grise
        frameDelta = cv2.absdiff(firstFrame, gray)
        cv2.imshow("4. Frame Delta", frameDelta)
        # Effectuer le seuillage de l'image pour transformer en blanc les pixels gris au delà de 25
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("5. Threshed Image",thresh)
        if cv2.countNonZero(thresh) != 0:
            text="Il y a du mouvement"
            color = (0, 0, 255)
            # Ecrire sur notre frame s'il y a du mouvement ou non
            cv2.putText(frame, text, (10, 20),cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        # Afficher le résultat
        cv2.imshow("6. Resulting Frame", frame)

        #Stocker l'image précédente pour pouvoir la comparer avec la suivante
        firstFrame = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    save_webcam('output.avi', 30.0, mirror=True)


if __name__ == '__main__':
    main()
