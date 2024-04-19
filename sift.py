import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import json
import sys

input = sys.argv[1]

query_img = cv.imread(input,cv.IMREAD_GRAYSCALE)

best_matches = 0
best_match = None
best_match_filename = ''
#store all this so we don't have to recalculate for plotting
best_kp1 = None
best_kp2 = None
best_good = []


for filename in os.listdir('luke_images'): 
    check_img = cv.imread(('luke_images/' + filename),cv.IMREAD_GRAYSCALE)
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query_img,None)
    kp2, des2 = sift.detectAndCompute(check_img,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.

    if len(good) > best_matches:
        best_matches = len(good)
        best_match = check_img
        best_match_filename = filename
        best_kp1 = kp1
        best_kp2 = kp2
        best_good = good

f = open('data/move_data_luke.json')
moves = json.load(f)

on_hit = ''
on_block = ''
name = ''

for m in moves:
    if m['file_name'] == best_match_filename:
        on_hit = m['on_hit']
        on_block = m['on_block']
        name = m['name']
        #print(m['name'])
        #print(f"On Hit: {on_hit}    On Block: {on_block}")
        break;

img3 = cv.drawMatchesKnn(query_img,best_kp1,best_match,best_kp2,best_good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.title(name)
plt.xlabel(f"On Hit: {on_hit}    On Block: {on_block}")
plt.show()