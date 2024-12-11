import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


# Show image
def show_image(title, image):
    if image.size == 0:
        print("image non-exists")
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

def initialize():
    if os.name == 'nt':
        os.system('del /f /q plates\\* chars\\*')
    elif os.name == 'posix':
        os.system('rm -rf plates/* chars/*')
    else:
        pass


class Locator:
    def __init__(self, path):
        self.path = path
        self.max_length = 700
        self.min_area = 2000

    # Uniform iamge size
    def zoom(self, w, h):
        widthScale = self.max_length / w
        heightScale = self.max_length / h

        scale = min(widthScale, heightScale)

        resizedWidth = int(w * scale)
        resizedHeight = int(h * scale)

        return resizedWidth, resizedHeight

    def get_accurate_plate(self, img_hsv, thres1, thres2, colour):
        rows,cols = img_hsv.shape[:2]
        left = cols
        right = 0
        top = rows
        bottom = 0

        rows_thres = rows * 0.8 if colour != "green" else rows * 0.5
        cols_thres = cols * 0.8 if colour != "green" else cols * 0.5

        for row in range(rows):
            count = 0
            for col in range(cols):
                H = img_hsv.item(row, col, 0)
                S = img_hsv.item(row, col, 1)
                V = img_hsv.item(row, col, 2)
                if thres1 < H <= thres2 and S > 34 and V > 46:
                    count += 1
            if count > cols_thres:
                if top > row:
                    top = row
                if bottom < row:
                    bottom = row
        for col in range(cols):
            count = 0
            for row in range(rows):
                H = img_hsv.item(row, col, 0)
                S = img_hsv.item(row, col, 1)
                V = img_hsv.item(row, col, 2)
                if thres1 < H <= thres2 and S > 34 and V > 46:
                    count += 1
            if count > rows_thres:
                if left > col:
                    left = col
                if right < col:
                    right = col
        return left, right, top, bottom

    def get_by_colour(self, adjusted_plates):
        colours = []
        for index, plate in enumerate(adjusted_plates):
            green = yellow = blue = 0
            img_hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
            rows, cols = img_hsv.shape[:2]
            size = rows * cols
            colour = None
            # count colours by pixels to determine the colour of the plate
            for row in range(rows):
                for col in range(cols):
                    H = img_hsv.item(row, col, 0)
                    S = img_hsv.item(row, col, 1)
                    V = img_hsv.item(row, col, 2)

                    if 11 < H <= 34 and S > 34:
                        yellow += 1
                    elif 35 < H <= 99 and S > 34:
                        green += 1
                    elif 99 < H <= 124 and S > 34:
                        blue += 1
            
            thres1 = thres2 = 0
            if yellow * 3 >= size:
                colour = "yellow"
                thres1 = 11
                thres2 = 34
            elif green * 3 >= size:
                colour = "green"
                thres1 = 35
                thres2 = 99
            elif blue * 3 >= size:
                colour = "blue"
                thres1 = 100
                thres2 = 124
            print(colour)
            colours.append(colour)
            # next plate
            if colour is None:
                adjusted_plates[index] = None
                continue
            left, right, top, bottom = self.get_accurate_plate(img_hsv, thres1, thres2, colour)
            w = right - left
            h = bottom - top
            if left == right or top == bottom:
                continue
            aspect_ratio = w / h
            if aspect_ratio < 2 or aspect_ratio > 4:
                continue

            # print("Coordinates of plate: {} {} {} {}".format(top, bottom, left, right))
            accurate_plate = plate[top:bottom, left:right]
            if accurate_plate.size != 0:
                adjusted_plates[index] = accurate_plate
                show_image("Accurate plate", accurate_plate)
    
    def affine(self, plates, vehicle_image, width, height):
        adjusted_plates = []
        for index, plate in enumerate(plates):
            if plate[2] > -1 and plate[2] < 1:
                angle = 1
            else:
                angle = plate[2]
            plate = (plate[0], (plate[1][0]+10, plate[1][1]+10), angle)
            box = cv2.boxPoints(plate)
            # get all coordinates
            w, h = plate[1][0], plate[1][1]
            if w > h:
                LT = box[1]
                LB = box[0]
                RT = box[2]
                RB = box[3]
            else:
                LT = box[0]
                LB = box[3]
                RT = box[1]
                RB = box[2]
    
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
            if affined_plate.size != 0:
                adjusted_plates.append(affined_plate)
                show_image("Affined plate", affined_plate)
        return adjusted_plates

    def projection_transform(self, plates, vehicle_image, width, height):
        adjusted_plates = []
        for index, plate in enumerate(plates):
            if plate[2] > -1 and plate[2] < 1:
                angle = 1
            else:
                angle = plate[2]
            plate = (plate[0], (plate[1][0]+10, plate[1][1]+10), angle)
            box = cv2.boxPoints(plate)
            # get all coordinates
            w, h = plate[1][0], plate[1][1]
            w, h = int(w), int(h)
            if w > h:
                LT = box[1]
                LB = box[0]
                RT = box[2]
                RB = box[3]
            else:
                w, h = h, w
                LT = box[0]
                LB = box[3]
                RT = box[1]
                RB = box[2]
    
            for point in [LT, LB, RT, RB]:
                pointLimit(point, width, height)

            # Projection transform
            src_points = np.float32([LT, RT, RB, LB])
            dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            adjusted_plate = cv2.warpPerspective(vehicle_image, M, (w, h))
            if adjusted_plate.size != 0:
                adjusted_plates.append(adjusted_plate)
                show_image("Adjusted plate", adjusted_plate)
        return adjusted_plates

    def find_plate(self):
        vehicle_image = cv2.imread(self.path)
        initialize()
        # show_image('Car Plate', vehicle_image)
        # Resize
        h, w = vehicle_image.shape[:2]
        width, height = self.zoom(w, h)
        vehicle_image = cv2.resize(vehicle_image, (width, height), interpolation=cv2.INTER_AREA)
        # Gaussian filter
        vehicle_image = cv2.GaussianBlur(vehicle_image, (7, 7), 0)
        gray_image = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
        # plt_show_gray(gray_image)
        # Open
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
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
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 19))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, se2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, se2)
        plt_show_gray(edges)

        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > self.min_area]
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
        # Affine transform
        adjusted_plates = self.projection_transform(plates, vehicle_image, width, height)
        print("Numbers of adjusted plates {}".format(len(adjusted_plates)))
        if len(adjusted_plates) > 0:
            self.get_by_colour(adjusted_plates)
            for i in range(len(adjusted_plates)):
                if adjusted_plates[i] is None:
                    continue 
                cv2.imwrite("plates/plate{}.jpg".format(i), adjusted_plates[i])
            return adjusted_plates
        return []


if __name__ == "__main__":
    locator = Locator("image/4.jpeg")
    locator.find_plate()
