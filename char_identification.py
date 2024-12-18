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
                          "吉", "冀", "津", "晋", "京", "辽", "鲁", "蒙", "闽", "宁", \
                          "青", "琼", "陕", "苏", "皖", "湘", "新", "渝", "豫", "粤", \
                          "云", "浙", "_"]
        self.template_w = 20
        self.template_h = 40

    def read_directory(self, path):
        templates = []
        directory_path = os.path.join("mainland_china_templates", path)
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            file_path = os.path.abspath(file_path)
            # opencv has trouble in resolving chinese characters, so I use numpy
            templates.append(file_path)
        return templates

    def read_templates(self):
        templates = []
        for i in self.template1:
            file_path = os.path.join("mainland_china_templates", i + ".jpg")
            file_path = os.path.abspath(file_path)
            template = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
            if template is None:
                print(f"Failed to read template: {i}")
                continue
            templates.append(file_path)
        return templates

    def identify_char(self):
        templates_list = []
        for i in self.template1:
            templates = self.read_directory(i)
            templates_list.append(templates)
        # templates = self.read_templates()
        w, h = self.template_w, self.template_h
        identified_plates = []
        for index, plate in enumerate(self.plates_in_chars):
            if plate is None:
                continue
            identified_chars = ""
            for j, char in enumerate(plate): 
                # plt_show_rgb(char)
                char = cv2.GaussianBlur((char), (3, 3), 0)
                gray_char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
                _, binary_char = cv2.threshold(gray_char, 0, 255, cv2.THRESH_OTSU)
                binary_char = cv2.resize(binary_char, (w, h))
                # unify the binary form for plates of all colours
                # area_white = 0
                # area_black = 0
                # for i in range(h):
                #     for k in range(w):
                #         if binary_char[i, k] == 255:
                #             area_white += 1
                #         else:
                #             area_black += 1
                # if area_white > area_black:
                #     _, binary_char = cv2.threshold(gray_char, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
                se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary_char = cv2.morphologyEx(binary_char, cv2.MORPH_CLOSE, se)
                cv2.imwrite(f"./output/{j}.jpg", binary_char)
                # plt_show_gray(binary_char)
                best_score = []
                for k, templates in enumerate(templates_list):
                    score = []
                    for template_path in templates:
                        template_img = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)
                        res = cv2.matchTemplate(binary_char, template_img, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        score.append([max_val, k])
                    score.sort(reverse=True)
                    best_score.append(score[0])
                best_score.sort(reverse=True)
                identified_char = self.template1[best_score[0][1]]
                if identified_char == "_":
                    identified_char = ""
                identified_chars += identified_char
            identified_plates.append(identified_chars)
            #     for k, template in enumerate(templates):
            #         res = cv2.matchTemplate(binary_char, template, cv2.TM_CCOEFF)
            #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            #         scores.append([res[0][0], k])
            #     scores.sort(reverse=True)
            #     identified_char = self.template1[scores[0][1]]
            #     identified_chars += identified_char
            # identified_plates.append(identified_chars)
        return identified_plates
        

if __name__ == "__main__":
    locator = Locator("image/4.jpeg")
    plates = locator.find_plate()
    segment = Segment(plates)
    plates_in_chars = segment.segment_plate()
    c = CharIdentification(plates_in_chars)
    chars = c.identify_char()
    print(chars)
    # for i in jing:
    #     plt_show_gray(i)
    