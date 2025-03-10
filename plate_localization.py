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
def plt_show_rgb(image, t=""):
    # swap channels
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image)
    plt.title(t)
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
        os.system('del /f /q plates\\* chars\\* output\\*')
    elif os.name == 'posix':
        os.system('rm -rf plates/* chars/* output/*')
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
        colour_plate = np.zeros((rows, cols), dtype=np.uint8)
        # black out the pixels that does not satisfy the colour of the plate
        for row in range(rows):
            for col in range(cols):
                H = img_hsv.item(row, col, 0)
                S = img_hsv.item(row, col, 1)
                V = img_hsv.item(row, col, 2)
                if colour == "black":
                    if V < 80:
                        colour_plate[row, col] = 255
                else:
                    if thres1 < H <= thres2 and S > 34 and V > 46:
                        colour_plate[row, col] = 255
        # use morphological operations to connect the white components and remove noise
        se_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        se_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        colour_plate = cv2.morphologyEx(colour_plate, cv2.MORPH_OPEN, se_open)
        colour_plate = cv2.morphologyEx(colour_plate, cv2.MORPH_CLOSE, se_close)
        # show_image("Colour plate", colour_plate)
        # find contours of the plate
        contours, _ = cv2.findContours(colour_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        # get convex hull (凸包)
        hull = cv2.convexHull(largest_contour)
        # approximate the quadrilateral shape of the plate
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx_quad = cv2.approxPolyDP(hull, epsilon, True)
        # divide the points into top-most points and bottom-most points
        sorted_points = sorted(approx_quad.reshape(-1, 2), key=lambda p: (p[1], p[0]))
        top = sorted_points[:2]  # top-most points
        bottom = sorted_points[-2:]  # bottom-most points
        # further divide two groups into left-most and right-most points
        left = sorted(top, key=lambda p: p[0]) 
        right = sorted(bottom, key=lambda p: p[0])
        # left top, right top, right bottom, left bottom
        approx_quad = np.array([left[0], left[1], right[1], right[0]])
        # display
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        cv2.drawContours(img_hsv, [approx_quad], -1, (0, 255, 0), 2)
        # show_image("Plate", img_hsv)
        return approx_quad

    def get_by_colour(self, adjusted_plates):
        colours = []
        for index, plate in enumerate(adjusted_plates):
            green = yellow = blue = black = 0
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
                    elif V < 150:
                        black += 1
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
            elif black * 3 >= size:
                colour = "black"
                thres1 = thres2 = 0
            print(colour)
            colours.append(colour)
            # next plate
            if colour is None:
                adjusted_plates[index] = None
                continue
            src_points = self.get_accurate_plate(img_hsv, thres1, thres2, colour)
            if src_points is None:
                adjusted_plates[index] = None
                continue
            tl, tr, br, bl = src_points
            # print("Coordinates of plate: {} {} {} {}".format(tl, tr, br, bl))
            accurate_plate = self.projection_transform(plate, src_points)
            if adjusted_plates[index] is not None:
                adjusted_plates[index] = [colour, accurate_plate]
    
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
            # show_image("Affined", affined_image)
            # Store affined plate
            affined_plate = affined_image[int(LT[1]):int(new_LB[1]), int(new_LB[0]):int(new_RB[0])]
            if affined_plate.size != 0:
                adjusted_plates.append(affined_plate)
                # plt_show_rgb(affined_plate, "Affine Transform")
        return adjusted_plates

    def projection_transform(self, plate, src_points):
        h, w = plate.shape[:2]
        h, w = int(h), int(w)
        # print("Width: {}, Height: {}".format(w, h))
        # Projection transform
        src_points = np.float32(src_points)
        dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        adjusted_plate = cv2.warpPerspective(plate, M, (w, h))
        if adjusted_plate.size != 0:
            # show_image("Adjusted plate", adjusted_plate)
            # plt_show_rgb(adjusted_plate, "Projection Transform")
            return adjusted_plate

    def find_plate(self):
        vehicle_image = cv2.imread(self.path)
        initialize()
        # show_image('Car Plate', vehicle_image)
        # Resize
        h, w = vehicle_image.shape[:2]
        width, height = self.zoom(w, h)
        vehicle_image = cv2.resize(vehicle_image, (width, height), interpolation=cv2.INTER_AREA)
        # Gaussian filter
        filtered_image = cv2.GaussianBlur(vehicle_image, (7, 7), 0)
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        # plt_show_gray(gray_image)
        # Open
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        opened_img = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, se1)
        # plt_show_gray(opened_img)
        # Weighted
        weighted_img = cv2.addWeighted(gray_image, 1, opened_img, -1, 0)
        # Binarize
        ret, binary_img = cv2.threshold(weighted_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary_img, 100, 200)
        # plt_show_gray(edges)
        # Close and open
        se_close = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
        se_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, se_close)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, se_open)
        # plt_show_gray(edges)

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
            # filter out invalid contours
            if aspect_ratio > 2 and aspect_ratio < 7:
                plates.append(rect)
                cv2.drawContours(canvas, contours, index, (255, 255, 255), 1, 8)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
        # show_image("image", canvas)
        print("Plates detected: {}".format(len(plates)))
        # Affine transform
        adjusted_plates = self.affine(plates, vehicle_image, width, height)
        print("Numbers of adjusted plates {}".format(len(adjusted_plates)))
        if len(adjusted_plates) > 0:
            self.get_by_colour(adjusted_plates)
            for i in range(len(adjusted_plates)):
                if adjusted_plates[i] is None:
                    continue
                _, adjusted_plate = adjusted_plates[i] 
                cv2.imwrite("plates/plate{}.jpg".format(i), adjusted_plate)
            return adjusted_plates
        return []


if __name__ == "__main__":
    locator = Locator("image/8.jpeg")
    locator.find_plate()
