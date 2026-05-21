import cv2
import numpy as np

img = cv2.imread("/Users/nathan/Documents/GitHub/MusicTransposer/IMG_2326.jpg", 0)
if img is None:
    raise FileNotFoundError("Image not found")
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = img.astype(np.uint8)

def process(block_size, c, word_w, word_h, para_h, min_area, max_area):
    # block_size must be odd and >= 3
    block_size = max(3, block_size | 1)

    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c
    )

    kernel_clean = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)

    kernel_word = np.ones((word_h, word_w), np.uint8)
    connected = cv2.dilate(cleaned, kernel_word, iterations=1)

    kernel_para = np.ones((para_h, 1), np.uint8)
    connected = cv2.dilate(connected, kernel_para, iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / max(h, 1)
        if area < min_area or area > max_area:
            continue
        if aspect_ratio < 0.1:
            continue
        if h > img.shape[0] * 0.5:
            continue
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return output

def nothing(x): pass

cv2.namedWindow("Tuner")
cv2.createTrackbar("block_size", "Tuner", 15,  51,  nothing)  # adaptive threshold region
cv2.createTrackbar("C",          "Tuner", 10,  30,  nothing)  # threshold aggressiveness
cv2.createTrackbar("word_w",     "Tuner", 15,  60,  nothing)  # horizontal char merging
cv2.createTrackbar("word_h",     "Tuner", 2,   10,  nothing)  # vertical char merging
cv2.createTrackbar("para_h",     "Tuner", 5,   30,  nothing)  # line merging
cv2.createTrackbar("min_area",   "Tuner", 100, 2000,nothing)  # min region size
cv2.createTrackbar("max_area",   "Tuner", 500, 1000,nothing)  # max region size (x100)

while True:
    block_size = cv2.getTrackbarPos("block_size", "Tuner")
    c          = cv2.getTrackbarPos("C",          "Tuner")
    word_w     = cv2.getTrackbarPos("word_w",     "Tuner")
    word_h     = cv2.getTrackbarPos("word_h",     "Tuner")
    para_h     = cv2.getTrackbarPos("para_h",     "Tuner")
    min_area   = cv2.getTrackbarPos("min_area",   "Tuner")
    max_area   = cv2.getTrackbarPos("max_area",   "Tuner") * 100  # scale up

    result = process(block_size, c, word_w, word_h, para_h, min_area, max_area)
    cv2.imshow("Tuner", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Final values — block_size:{block_size} C:{c} word_w:{word_w} word_h:{word_h} para_h:{para_h} min_area:{min_area} max_area:{max_area}")
        break

cv2.destroyAllWindows()