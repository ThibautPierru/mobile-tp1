import cv2

diff = 0
plus_ressemblant = ""
nom_image_ref = 'Data-TP/waves.jpg'
image_ref = cv2.imread(nom_image_ref,0)
hist_image_ref = cv2.calcHist([image_ref],[0],None,[256],[0,256])
cv2.normalize(hist_image_ref,hist_image_ref,0,1,cv2.NORM_MINMAX)
liste_image = ["Data-TP/beach.jpg","Data-TP/dog.jpg","Data-TP/polar.jpg","Data-TP/bear.jpg","Data-TP/lake.jpg","Data-TP/moose.jpg"]
for image in liste_image:
  current_image = cv2.imread(image,0)
  hist_current_image = cv2.calcHist([current_image],[0],None,[256],[0,256])
  cv2.normalize(hist_current_image, hist_current_image, 0, 1, cv2.NORM_MINMAX)
  diff_hist = cv2.compareHist(hist_image_ref,hist_current_image,cv2.HISTCMP_CORREL)
  print("Différence entre l'image {} et {} au niveau de l'histogramme : {}".format(nom_image_ref,image,diff_hist))
  if diff < diff_hist:
   diff = diff_hist
   plus_ressemblant = image

print("L'image la plus ressemblante de {} est : {}".format(nom_image_ref,plus_ressemblant))
