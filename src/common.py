import numpy as np
import cv2

IMG_H, IMG_W = 84, 84
N_ACTIONS = 4

def processState(state1, img_h = IMG_H, img_w = IMG_W):
    greyscale = np.asarray(state1).sum(axis=2)/3
    cropped = greyscale[17:, :]
    scaled = cv2.resize(cropped, dsize=(img_h, img_w), interpolation=cv2.INTER_CUBIC)
    
    return np.reshape(scaled,[img_h * img_w])

'''
Break a list into equally sized Chunks
'''
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
