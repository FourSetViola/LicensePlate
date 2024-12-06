import cv2
import numpy as np
from matplotlib import pyplot as plt

# Show image
def show_image(title, image):
    if image.size == 0:
        return
    cv2.imshow(title, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Plt show image
def plt_show_rgb(image):
    # swap channels
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image)
    plt.show()

# Plt show gray image
def plt_show_gray(image):
    plt.imshow(image, cmap='gray')
    plt.show()

# Uniform iamge size
def zoom(w, h, wMax, hMax):
    widthScale = wMax / w
    heightScale = hMax / h

    scale = min(widthScale, heightScale)

    resizedWidth = int(w * scale)
    resizedHeight = int(h * scale)

    return resizedWidth, resizedHeight

vehicle_image = cv2.imread('image/3.jpg')
# show_image('Car Plate', vehicle_image)
# Predefined parameters
maxLength = 700 # Maximum length of resized image
min_area = 2000 # Minimum area for a component to be considered as a possible plate
# Resize
h, w = vehicle_image.shape[:2]
width, height = zoom(w, h, maxLength, maxLength)
vehicle_image = cv2.resize(vehicle_image, (width, height))
# Gaussian filter
vehicle_image = cv2.GaussianBlur(vehicle_image, (3, 3), 0)
gray_image = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
# plt_show_gray(gray_image)
# Open
se1 = np.ones((20, 20), np.uint8)
opened_img = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, se1)
# plt_show_gray(opened_img)
# Weighted
weighted_img = cv2.addWeighted(gray_image, 1, opened_img, -1, 0)
# Binarize
ret, binary_img = cv2.threshold(weighted_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plt_show_gray(binary_img)
# Find edge
edges = cv2.Canny(binary_img, 100, 200)
# plt_show_gray(edges)
# Close and open
se2 = np.ones((10, 19), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, se2)
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, se2)
plt_show_gray(edges)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
# Delete invalid contours
plates = []
canvas = vehicle_image.copy()
for index, contour in enumerate(contours):
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w < h:
        w, h = h, w # swap width and height
    aspect_ratio = w / h
    if aspect_ratio > 2 and aspect_ratio < 4:
        plates.append(rect)
        cv2.drawContours(canvas, contours, index, (255, 255, 255), 1, 8)
        
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(canvas, [box], 0, (0, 0, 255), 1)
show_image("image", canvas)
print("Plates detected: {}".format(len(plates)))

# Rect rotation
def pointLimit(point, maxWidth, maxHeight):
    if point[0] < 0:
        point[0] = 0
    if point[0] > maxWidth:
        point[0] = maxWidth
    if point[1] < 0:
        point[1] = 0
    if point[1] > maxHeight:
        point[1] = maxHeight

rotated_plates = []
for index, plate in enumerate(plates):
    if plate[2] > -1 and plate[2] < 1:
        angle = 1
    else:
        angle = plate[2]
    plate = (plate[0], (plate[1][0]+5, plate[1][1]+5), angle)
    box = cv2.boxPoints(plate)
    # get all coordinates
    w, h = plate[1][0], plate[1][1]
    if w > h:
        LT = box[1]
        LB = box[0]
        RT = box[2]
        RB = box[3]
    else:
        LT = box[2]
        LB = box[1]
        RT = box[3]
        RB = box[0]
    
    for point in [LT, LB, RT, RB]:
        pointLimit(point, width, height)
    
    # Affine
    new_LB = [LT[0], LB[1]]
    new_RB = [RB[0], LB[1]]
    old_triangle = np.float32([LT, LB, RB])
    new_triangle = np.float32([LT, new_LB, new_RB])
    warpMat = cv2.getAffineTransform(old_triangle, new_triangle)
    affined_image = cv2.warpAffine(vehicle_image, warpMat, (width, height))
    show_image("Affined", affined_image)
    # Store affined plate
    affined_plate = affined_image[int(LT[1]):int(new_LB[1]), int(new_LB[0]):int(new_RB[0])]
    rotated_plates.append(affined_plate)
    print(index)
    print("LT", LT)
    print("LB", LB)
    print("RB", RB)
    show_image("Affined plate", affined_plate)
