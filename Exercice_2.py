import cv2
import numpy as np



def main():
    img_base = cv2.imread('Data-TP/bgr.png')

    blues,greens,reds = cv2.split(img_base)
    channel_vide = np.zeros(img_base.shape[:2], np.uint8)
    reds = cv2.merge((channel_vide,channel_vide,reds))
    blues = cv2.merge((blues,channel_vide,channel_vide))
    greens = cv2.merge((channel_vide,greens,channel_vide))
    bg = cv2.add(blues, greens)
    br = cv2.add(blues, reds)
    rg = cv2.add(reds, greens)

    cv2.imshow('Image de base',img_base)
    cv2.imshow('Blues',blues)
    cv2.imshow('Greens',greens)
    cv2.imshow('Rouge',reds)
    cv2.imshow('B + G', bg)
    cv2.imshow('B + R', br)
    cv2.imshow('R + G', rg)
    if cv2.waitKey(0) & 0xFF == ord('q'):  # if 'q' is pressed then quit
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()