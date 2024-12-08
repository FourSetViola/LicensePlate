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


class Segment:
    def __init__(self, plates=[]):
        self.plates = plates

    def segment_plate(self):
        for index, plate in enumerate(self.plates):
            # grayscale
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
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary_plate = cv2.morphologyEx(binary_plate, cv2.MORPH_OPEN, se)
            show_image("dilated binary_plate", binary_plate)
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
            i = 0
            print("chars: ", len(chars))
            for char in chars:
                x, y, w, h = char
                aspect_ratio = h / w
                if (aspect_ratio > 1.8) and (aspect_ratio < 2.5):
                    i += 1
                    char_image = plate[y:y + h, x:x + w]
                    # show_image("char{}".format(i), char_image)
                    cv2.imwrite("chars/plate{}_char{}.jpg".format(index,i), char_image)

if __name__ == "__main__":
    from plate_localization import Locator
    locator = Locator("image/4.jpg")
    plates = locator.find_plate()
    segment = Segment(plates)
    segment.segment_plate()
