import cv2
from hand_finder import find_hand
from string import ascii_lowercase
import numpy as np
import binascii
import os
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for i in ascii_lowercase:
    if i != "j" and i != "z" :
        chars.append(i)

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

current = 0

prev_cnt = None

def get_sample(pos):
    return cv2.resize(cv2.imread("./samples/"+chars[pos]+".jpg"),(400,400))

current_sample = get_sample(current)
hand_frames = []
while True:
    ret, frame = cam.read()
    #cv2.imshow("test", frame)
    if ret:
        cur_cnt, hand_frame = find_hand(frame)
        if hand_frame.shape[0] != 400 and hand_frame.shape[1] != 400:
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
            img_name = "./dataset/newsample1/"+chars[current]+"/opencv_frame_{}.png".format(binascii.b2a_hex(os.urandom(15)))
            cv2.imwrite(img_name, hand_frame)
            print("{} written!".format(img_name))
            img_counter += 1
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