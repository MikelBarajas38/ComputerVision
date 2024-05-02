import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse

# images (assuming svm was pretraied)
gt_images = ['img/train/aal/original.png', 'img/train/op/original.png', 'img/train/tomats/original.png']

# labels

labels = {
    1: 'All About Love',
    2: 'One Piece',
    3: 'El Viejo y la Mar'
}

colors = {
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 0, 0)
}

# classifiers

sift = cv2.SIFT_create()
sift_features = []

orb = cv2.ORB_create()
orb_features = []

win_size = (64, 128)
block_size = (32, 32)
block_stride = (16, 16)
cell_size = (16, 16)
nbins = 9   

hog =  cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
svm = cv2.ml.SVM_load('svm_model_final.yml')

def init():
    for img_path in gt_images:
    
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des1 = sift.detectAndCompute(gray, None)
        sift_features.append(des1)

        kp, des2 = orb.detectAndCompute(gray, None)
        orb_features.append(des2)

# utils

def find_best_match(des1, des2, norm_type = cv2.NORM_L2):
    #bf = cv2.BFMatcher(normType=norm_type, crossCheck=True)
    bf = cv2.BFMatcher(normType=norm_type)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return len(good)

# detection and classification

def detect_and_classify_sift(img, thresh = 20):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    
    best_match = 0
    best_match_index = -1

    for i in range(len(sift_features)):
        match = find_best_match(des, sift_features[i])

        if match > best_match and match > thresh:
            best_match = match
            best_match_index = i + 1
    
    return best_match_index

def detect_and_classify_orb(img, thresh = 5):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    
    best_match = 0
    best_match_index = 0

    for i in range(len(orb_features)):
        match = find_best_match(des, orb_features[i], cv2.NORM_HAMMING)
        if match > best_match and match > thresh:
            best_match = match
            best_match_index = i + 1
    
    return best_match_index

def detect_and_classify_hog_auto(img):

    svm = cv2.ml.SVM_load('svm_model_final.yml')

    hog.setSVMDetector(svm.getSupportVectors()[0])
 
    locations, scores = hog.detectMultiScale(img)
    x, y, w, h = locations[np.argmax(scores.flatten())]
    window = img[y:y+h, x:x+w]
    window = cv2.resize(window, win_size)

    features = hog.compute(window).flatten()
    pred = svm.predict(features.reshape(1, -1))[1][0][0]
    
    return int(pred)

def detect_and_classify_hog(img):
 
    window = img
    window = cv2.resize(window, win_size)

    features = hog.compute(window).flatten()
    pred = svm.predict(features.reshape(1, -1))[1][0][0]
    
    return int(pred)

def detect_and_classify_hog_auto(img):

    hog.setSVMDetector(svm.getSupportVectors()[0])
 
    locations, scores = hog.detectMultiScale(img)
    x, y, w, h = locations[np.argmax(scores.flatten())]
    window = img[y:y+h, x:x+w]
    window = cv2.resize(window, win_size)

    features = hog.compute(window).flatten()
    pred = svm.predict(features.reshape(1, -1))[1][0][0]
    
    return int(pred)


def detect_and_classify(frame, classifier):
    if classifier == 'sift':
        return detect_and_classify_sift(frame)
    elif classifier == 'orb':
        return detect_and_classify_orb(frame)
    elif classifier == 'hog':
        return detect_and_classify_hog_auto(frame)
    else:
        print(f'Error: Invalid classifier "{classifier}"')
        return None


def main(classifier):
    
    init()
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        pred = detect_and_classify(frame, classifier)

        if pred is not None and pred > 0:
            print(f'Prediction: {labels[pred]}')
            label_text = labels[pred]
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[pred], 2, cv2.LINE_AA)

        cv2.imshow(f'{classifier}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object detection and classification from live video feed')
    parser.add_argument('--classifier', type=str, choices=['sift', 'orb', 'hog'], default='sift',
                        help='Choose the classifier to use (sift, orb, hog)')
    args = parser.parse_args()
    main(args.classifier)