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
        self.templates = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
                          "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", \
                          "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", \
                          "W", "X", "Y", "Z", \
                          "藏", "川", "鄂", "甘", "赣", "港", "贵", "桂", "黑", "沪", \
                          "吉", "冀", "津", "晋", "京", "辽", "鲁", "蒙", "闽", "宁", \
                          "青", "琼", "陕", "苏", "皖", "湘", "新", "渝", "豫", "粤", \
                          "云", "浙", "_"]
        self.template_w = 20
        self.template_h = 40

    def read_directory(self, path):
        templates = []
        directory_path = os.path.join("templates", path)
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            file_path = os.path.abspath(file_path)
            # opencv has trouble in resolving chinese characters, so I use numpy
            templates.append(file_path)
        return templates

    def read_templates(self):
        templates = []
        for i in self.templates:
            file_path = os.path.join("templates", i + ".jpg")
            file_path = os.path.abspath(file_path)
            template = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            if template is None:
                print(f"Failed to read template: {i}")
                continue
            templates.append(file_path)
        return templates

    def identify_char(self):
        templates_list = []
        for i in self.templates:
            templates = self.read_directory(i)
            templates_list.append(templates)
        # templates = self.read_templates()
        w, h = self.template_w, self.template_h
        identified_plates = []
        for index, plate in enumerate(self.plates_in_chars):
            if plate is None:
                continue
            identified_chars = ""
            colour, plate = plate
            for j, char in enumerate(plate): 
                # plt_show_rgb(char)
                char = cv2.resize(char, (w, h))
                char = cv2.GaussianBlur((char), (1, 1), 0)
                gray_char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
                if colour == "blue" or colour == "black":
                    _, binary_char = cv2.threshold(gray_char, 0, 255, cv2.THRESH_OTSU)
                else:
                    _, binary_char = cv2.threshold(gray_char, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary_char = cv2.morphologyEx(binary_char, cv2.MORPH_CLOSE, se)
                cv2.imwrite(f"./output/{j}.jpg", binary_char)
                # plt_show_gray(binary_char)
                best_score = []
                # loop through all templates of different characters
                for k, templates in enumerate(templates_list):
                    score = []
                    # loop through all templates of the same character
                    for template_path in templates:
                        template_img = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)
                        res = cv2.matchTemplate(binary_char, template_img, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        score.append([max_val, k])
                    # scores for the same character
                    score.sort(reverse=True)
                    best_score.append(score[0])
                # best matching character
                best_score.sort(reverse=True)
                identified_char = self.templates[best_score[0][1]]
                if identified_char == "_":
                    identified_char = ""
                identified_chars += identified_char
            if identified_chars != "":
                identified_plates.append(identified_chars)
        return identified_plates
        

if __name__ == "__main__":
    locator = Locator("image/4.jpeg")
    plates = locator.find_plate()
    segment = Segment(plates)
    plates_in_chars = segment.segment_plate()
    c = CharIdentification(plates_in_chars)
    chars = c.identify_char()
    print(chars)
    