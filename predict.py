import sys
chars = []
if len(sys.argv) <= 1:
    print "Please provide \n--digits to load model to predict ASL digits\n--alphabets to load model to predict ASL alphabets"
    exit()
if sys.argv[1] == "--digits":
    for i in range(10):
        chars.append(str(i))
    from keras.models import load_model
    model = load_model("./digits_97_weights.hdf5")
elif sys.argv[1] == "--alphabets":
    from string import ascii_lowercase
    for i in ascii_lowercase:
        if i != "j" and i != "z" :
            chars.append(i)
    from keras.models import load_model
    model = load_model("./alphabets_80_weights.hdf5")
else:
    print "Please provide \n--digits to load model to predict ASL digits\n--alphabets to load model to predict ASL alphabets"
    exit()

import cv2
from hand_finder import find_hand
import numpy as np
import binascii
import os


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

current = 0

prev_cnt = None

def get_sample(pos):
    return cv2.resize(cv2.imread("./samples/"+chars[pos]+".jpg"),(200,200))

current_sample = get_sample(current)
    
while True:
    ret, frame = cam.read()
    if ret:
        cur_cnt, hand_frame = find_hand(frame)
        if hand_frame.shape[0] != 200 and hand_frame.shape[1] != 200:
            continue
        if cur_cnt is not None:
            # Removing Noise from Hand Frame
            show_frame = np.concatenate((current_sample, hand_frame), axis=1)

            cv2.imshow("test", show_frame)

            if type(prev_cnt) == 'NoneType':
                prev_cnt = cur_cnt
            
            match_percentage = cv2.matchShapes(cur_cnt, prev_cnt, 2, 0.0)
            prev_cnt = cur_cnt
            if match_percentage > 0.7:
                prev_cnt = cur_cnt
        else:
            continue
        k = cv2.waitKey(50)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k%256 == 32:    
            # SPACE pressed
            if hand_frame.shape[0] == 200 and hand_frame.shape[1] ==200:
                # Evaluate the model and print results
                prediction = model.predict(np.array([hand_frame]),batch_size=1, verbose=1)
                current_sample = get_sample(np.argmax(prediction))

        elif k%256 == 3:
            current = (current + 1 ) % 34
            current_sample = get_sample(current)
            img_counter = 0
        elif k%256 == 2:
            current = (current - 1 ) % 34
            if current < 0:
                current = 33
            current_sample = get_sample(current)
            img_counter = 0
    else:
        continue

cam.release()

cv2.destroyAllWindows()