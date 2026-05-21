import cv2 


img = cv2.imread("IMG_2326.jpg", 0)

#img = cv2.resize(img, (360, 480))

print(img.shape)

cv2.imshow("imagef", img)
def pre_processing(img):
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.Canny(img, 25, 75)
    return img




img = pre_processing(img)
cv2.imshow("image", img)


cv2.waitKey(0)