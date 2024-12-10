import cv2
from plate_localization import Locator
from plate_segmentation import Segment
import matplotlib.pyplot as plt
import numpy as np
import os


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


class CharIdentification:
    def __init__(self, plates_in_chars=[]):
        self.plates_in_chars = plates_in_chars
        self.template1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
                          "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", \
                          "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", \
                          "W", "X", "Y", "Z", \
                          "藏", "川", "鄂", "甘", "赣", "港", "贵", "桂", "黑", "沪", \
                          "吉", "冀", "津", "晋", "jing", "辽", "鲁", "蒙", "闽", "宁", \
                          "青", "琼", "陕", "苏", "皖", "湘", "新", "渝", "豫", "粤", \
                          "云", "浙"]

    def read_templates(self):
        templates = []
        for i in self.template1:
            file_path = "mainland_china_templates/{}.jpg".format(i)
            file_path = file_path.encode("utf-8").decode("utf-8")
            template = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                # print(i)
                continue
            templates.append(template)
        return templates
    
    def identify_char(self):
        templates = self.read_templates()
        for index, plate in enumerate(self.plates_in_chars):
            for j, char in enumerate(plate):
                plt_show_rgb(char)
                char = cv2.GaussianBlur((char), (3, 3), 0)
                gray_char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
                _, binary_char = cv2.threshold(gray_char, 0, 255, cv2.THRESH_OTSU)
                plt_show_gray(binary_char)
                for k, template in enumerate(templates):
                    w, h = template.shape[::-1]
                    binary_char = cv2.resize(binary_char, (w, h))
                    res = cv2.matchTemplate(binary_char, template, cv2.TM_CCOEFF_NORMED)



if __name__ == "__main__":
    test_char = cv2.imread("chars/plate0_char2.jpg")
    c = CharIdentification([[test_char]])
    c.identify_char()
    # img = cv2.imread("mainland_china_templates/A.jpg", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("img", img)
    # cv2.waitKey()