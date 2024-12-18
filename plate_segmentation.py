import matplotlib.pyplot as plt
import cv2


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


def zoom(w, h, max_length):
        widthScale = max_length / w
        heightScale = max_length / h

        scale = min(widthScale, heightScale)

        resizedWidth = int(w * scale)
        resizedHeight = int(h * scale)

        return resizedWidth, resizedHeight

class Segment:
    def __init__(self, plates=[]):
        self.plates = plates

    def segment_plate(self):
        plates_in_chars = []
        for index, plate in enumerate(self.plates):
            plate_in_chars = []
            # grayscale
            if plate is None:
                continue
            h, w = plate.shape[:2]
            resizedWidth, resizedHeight = zoom(w, h, 250)
            plate = cv2.resize(plate, (250, 80), interpolation=cv2.INTER_AREA)
            gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            # binarize
            _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_OTSU)
            # unify the binary form for plates of all colours
            area_white = 0
            area_black = 0
            h, w = binary_plate.shape
            for i in range(h):
                for j in range(w):
                    if binary_plate[i, j] == 255:
                        area_white += 1
                    else:
                        area_black += 1
            if area_white > area_black:
                _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            # show_image("unified binary_plate", binary_plate)
            # dilation
            # edges = cv2.Canny(binary_plate, 90, 100)
            # show_image("edges", edges)
            se_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, se_circle)
            show_image("De-noise", binary_plate)
            se_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, se_vertical)
            show_image("Connect Characters", binary_plate)
            # se_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            # binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, se_close)
            # binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_DILATE, se1)
            # show_image("dilate1", binary_plate)
            # binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_DILATE, se1)
            # show_image("dilate2", binary_plate)
            # binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, se2)
            show_image("Strengthen edges", binary_plate)
            # find contour
            contours, hierarchy = cv2.findContours(binary_plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [contour for contour in contours if cv2.contourArea(contour) > 300]
            # view contours
            plate_copy = plate.copy()
            cv2.drawContours(plate_copy, contours, -1, (0, 0, 255), 5)
            show_image("contours", plate_copy)
            # segment characters
            chars = []
            for contour in contours:
                char = []
                rect = cv2.boundingRect(contour)
                x, y, w, h = rect
                char.append(x)
                char.append(y)
                char.append(w)
                char.append(h)
                chars.append(char)
            chars = sorted(chars, key=lambda char: char[0])
            i = 1
            for char in chars:
                x, y, w, h = char
                aspect_ratio = h / w
                if (aspect_ratio > 1.8 and aspect_ratio < 3) or (aspect_ratio > 5 and aspect_ratio < 6) or (aspect_ratio > 7 and aspect_ratio < 8):
                    i += 1
                    char_image = plate[y:y + h, x:x + w]
                    plate_in_chars.append(char_image)
                    # show_image("char{}".format(i), char_image)
                    cv2.imwrite("chars/plate{}_char{}.jpg".format(index,i), char_image)
            if len(plate_in_chars) != 0:
                print("chars: ", len(chars))
                plates_in_chars.append(plate_in_chars)
                cv2.imwrite("./output/processed_plate.jpg", binary_plate)
                plt_show_gray(binary_plate)
        return plates_in_chars

if __name__ == "__main__":
    from plate_localization import Locator
    locator = Locator("image/4.jpeg")
    plates = locator.find_plate()
    segment = Segment(plates)
    segment.segment_plate()
