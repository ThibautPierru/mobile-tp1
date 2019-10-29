import cv2

pas = 1

img_horloge = cv2.imread("horloge.bmp")
hist_horloge = cv2.calcHist([img_horloge],[0],None,[256],[0,256])
cv2.normalize(hist_horloge,hist_horloge,0,1,cv2.NORM_MINMAX)
h_horloge,w_horloge = img_horloge.shape[:2]

img = cv2.imread("Data-TP/parliament3.bmp")
height, width = img.shape[:2]

maxdiff = 0
coordinate = dict()

w,h=0,0
while w < width:
    while h < height:
        tmp_img = img[h:h+h_horloge,w:w+w_horloge]
        tmp_hist = cv2.calcHist([tmp_img],[0],None,[256],[0,256])
        cv2.normalize(tmp_hist, tmp_hist, 0, 1, cv2.NORM_MINMAX)
        diff = cv2.compareHist(tmp_hist,hist_horloge,cv2.HISTCMP_CORREL)
        h+=pas
        if diff > maxdiff:
            maxdiff = diff
            coordinate = {"w":w,"h":h}
    h = 0
    w+=pas
print("Coordonnées :",coordinate)
w_affichage = coordinate.get("w")
h_affichage = coordinate.get("h")
diff = maxdiff
cv2.rectangle(img,(w_affichage,h_affichage),(w_affichage+w_horloge,h_affichage+h_horloge),(255,0,255),0)
cv2.imshow("Result", img)
print("Correspondance la plus grande : ",diff)
cv2.waitKey(0)
cv2.destroyAllWindows()