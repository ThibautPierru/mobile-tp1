import cv2

def calculate_histogram(img):
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
    return histogram

class KNeighborsClassifier(object):
    def __init__(self,n_neighbors=5,algorithm='brute'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm

    def fit(self,training_data,target_values):
        self.data = []
        for image, id_category in zip(training_data,target_values):
            self.data.append((id_category,calculate_histogram(image)))
        return self

    def predict(self,images_to_predict):
        if self.data is None:
            print("Veuillez d'abord entraîner l'algorithme")
            return
        if self.algorithm == 'brute':
            result = []
            for image in images_to_predict:
                list_compare_hist = []
                for id_category,histogram in self.data:
                    # Calculer les différences d'histogramme et les stocker dans une liste
                    diff_hist = cv2.compareHist(calculate_histogram(image),histogram,cv2.HISTCMP_CORREL)
                    list_compare_hist.append((id_category,diff_hist))
                # Trier la liste d'histogrammes pour avoir les images les plus ressemblantes au début
                list_compare_hist.sort(key=lambda list:list[1],reverse=True)
                predictions = dict()
                for i in range(self.n_neighbors):
                    """Prendre les N plus proches histogrammes pour les ajouter dans un dictionnaire afin de calculer
                    le nombre d'occurence de chaque catégorie"""
                    id_category = list_compare_hist[i][0]
                    if id_category in predictions:
                        predictions[id_category]+=1
                    else:
                        predictions[id_category] = 1
                # Trier le dictionnaire pour ressortir la catégorie dont les histogrammes ont été les plus présents
                sorted_predictions = sorted(predictions.items(),key=lambda kv:kv[1],reverse=True)
                print("Sorted predictions :",sorted_predictions)
                most_matching_label = sorted_predictions[0][0]
                result.append(most_matching_label)
            return result





