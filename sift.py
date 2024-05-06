import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import json
import sys
import glob
import scipy

#filter additions
sobel_x = np.array([
      [1, 0, -1],
      [2, 0, -2],
      [1, 0, -1]
])

sobel_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

def get_gradient_mag(image):
    Ix = scipy.signal.convolve(image, sobel_x, mode='same')
    Iy = scipy.signal.convolve(image, sobel_y, mode='same')
    image = np.sqrt(Iy**2 + Ix**2)
    return image

def gaussian_blur(image, filter_size, sigma):
    assert(filter_size % 2 == 1)
    h = (filter_size + 1)//2
    d = np.arange(h)
    d = np.concatenate((d[::-1], d[1:]))
    d = d[:, np.newaxis]
    d_sq = d**2 + d.T ** 2
    # Take the gaussian
    g = np.exp(-d_sq/2/(sigma**2))
    # Normalize
    g = g/g.sum().sum()
    image = scipy.signal.convolve(image, g, mode='same')
    return image


input = sys.argv[1]

query_img = cv.imread(input,cv.IMREAD_GRAYSCALE)

# ADD GAUSS BLUR
#query_img = gaussian_blur(query_img, 51, 3)
#cv.imwrite('query.png', query_img)
#query_img = cv.imread("query.png",cv.IMREAD_GRAYSCALE)

# ADD GRADIENT MAG
#query_img = get_gradient_mag(query_img)
#cv.imwrite('query.png', query_img)
#query_img = cv.imread("query.png",cv.IMREAD_GRAYSCALE)

best_matches = 0
best_match = None
best_match_filename = ''
#store all this so we don't have to recalculate for plotting
best_kp1 = None
best_kp2 = None
best_good = []

# store for later
on_hit = ''
on_block = ''
name = ''

print("Sifting through all the images...")
for root, dirs, files in os.walk("."):
    for directory in dirs:
        if directory.endswith("deejay_images"): #change this back
            dir_path = os.path.join(root, directory)

            image_files = glob.glob(os.path.join(dir_path, "*"))
    
            for image_file in image_files:
                check_img = cv.imread((image_file),cv.IMREAD_GRAYSCALE)

                # ADD GAUSS BLUR
                #check_img = gaussian_blur(check_img, 51, 3)
                #cv.imwrite('check.png', check_img)
                #check_img = cv.imread("check.png",cv.IMREAD_GRAYSCALE)

                # ADD GRADIENT MAG
                #check_img = get_gradient_mag(check_img)
                #cv.imwrite('check.png', check_img)
                #check_img = cv.imread("check.png",cv.IMREAD_GRAYSCALE)

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
                    if m.distance < 0.65*n.distance:
                        good.append([m])
                # cv.drawMatchesKnn expects list of lists as matches.

                if len(good) > best_matches:
                    best_matches = len(good)
                    best_match = check_img
                    best_match_filename = image_file
                    best_kp1 = kp1
                    best_kp2 = kp2
                    best_good = good


print("Reading through the json files to find match, and post the on hit and on block...")
for json_file in os.listdir("./data"):
    # Full path of the JSON file
    json_file_path = os.path.join("./", json_file)

    # remove path file
    best_match_filename = os.path.basename(best_match_filename)
    print(best_match_filename)
    
    # Process the JSON file
    with open("./data/" + json_file, 'r') as f:
        try:
            data = json.load(f)

            character = data.get("name")
            moves = data.get("moves", [])
            for move in moves:
                if move['file_name'] == best_match_filename:
                    print("entered")
                    print(move['on_hit'], move['on_block'], move['name'])
                    on_hit = move['on_hit']
                    on_block = move['on_block']
                    name = character + " " + move['name']
                    #print(m['name'])
                    #print(f"On Hit: {on_hit}    On Block: {on_block}")
                    break            
            # Do something with the JSON data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file '{json_file_path}': {e}")

print("image found!")
img3 = cv.drawMatchesKnn(query_img,best_kp1,best_match,best_kp2,best_good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.title(name)
print(on_hit)
print(on_block)
plt.xlabel(f"On Hit: {on_hit}    On Block: {on_block}")
plt.show()