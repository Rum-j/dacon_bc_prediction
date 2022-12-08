## crop image
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans 
import copy 

def crop_from(file_path, print_imgs = True ):

    if file_path is None:
        file_path= '/Users/krc/Documents/breast_dacon/train_imgs/'
    os.mkdir('new_train')
    file_list = os.walk(file_path)  # OS로 불러오기
    img_files = [file for file in file_list if file[-1][-1].endswith(".png")]

    score_dict = {}

    if not img_files:  # if empty folder
        print("there are no png files")
        sys.exit()

    for i, f in enumerate(img_files[0][2]):
        # t.ly/zgLP
        score_list = []
        image = cv2.imread(file_path + f)
        y_orig, x_orig, channel= image.shape
        if x_orig / y_orig < 1.4:
            print(f'index {i} has one image')
            cv2.imwrite(f'./new_train/{f}', image)
            score_dict[f] = copy.deepcopy(score_list)
            score_list.clear()
            continue
        gray_sample = image.copy()
        gray_sample = cv2.cvtColor(gray_sample, cv2.COLOR_RGB2GRAY)
        
        # 모양 맞추기 flip 
        gray_sample =cv2.flip(gray_sample, 0)
        # resize to 400 * 200
        gray_sample = cv2.resize(gray_sample, dsize=(400, 200), interpolation=cv2.INTER_LINEAR) 
        
        #전처리
        coord = np.where( gray_sample < 239 )
        co_array = np.array(coord)
        co_array = np.float32(co_array).T
        from sklearn.metrics import silhouette_samples, silhouette_score

        best_score = 0
        best_k = 1
        best_centroids = []

        for j, k in enumerate([ 4, 3, 2]):

            # Run the Kmeans algorithm
            km = KMeans(n_clusters=k)
            km.fit(co_array)
            labels = km.predict(co_array) # input data

            centroids = km.compute_centroids(co_array, labels) # cluster_centers_
            # Get silhouette samples
            silhouette_vals = silhouette_samples(co_array, labels)

            # Get the average silhouette score and plot it
            avg_score = np.mean(silhouette_vals) ## score 
            score_list.append(avg_score)
            print('K = ',k, 'avg_score:', avg_score)

            if best_score < avg_score:
                if best_k == 3 and k ==2 and (avg_score-best_score) < 0.09:
                    break
                best_score = avg_score
                best_k = k
                best_centroids = centroids
                x_coord = centroids[ : , 1]
                x_coord.sort()        
                if best_k == 2:
                    x1, x2 = x_coord
                elif best_k ==3  :
                    x1, x2, x3 = x_coord
                elif best_k ==4  :
                    x1, x2, x3, x4 = x_coord

        score_dict[f] = copy.deepcopy(score_list)
        score_list.clear()
        if best_k == 2:
            crop_image = image[:, : int( (x1+x2)/2  *x_orig / 400 ), :]
        elif best_k > 2:
            alpha = int(y_orig / 2)
            crop_image = image[:, max(0, int(x2 *x_orig /400) -alpha)  :int(x2 *x_orig /400) +alpha , : ] 
        cv2.imwrite(f'./new_train/{f}', crop_image)
            
        # print(f'image {f} beset centroids : ', best_centroids)
        if print_imgs == True:
            plt.title(f'{f} image result')
            plt.imshow(crop_image)
            plt.show()
    return score_dict
    