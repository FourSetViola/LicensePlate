import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from char_identification import CharIdentification
from plate_localization import Locator
from plate_segmentation import Segment
import cv2

class PlateRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车牌识别系统")
        self.setGeometry(100, 100, 800, 600)

        self.image_path = None

        self.label = QLabel("选择一张图片进行车牌识别", self)
        self.label.setGeometry(50, 50, 700, 50)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 100, 700, 400)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(200, 520, 550, 30)

        self.button = QPushButton("选择图片", self)
        self.button.setGeometry(50, 520, 100, 30)
        self.button.clicked.connect(self.load_image)

        self.button = QPushButton("识别车牌", self)
        self.button.setGeometry(650, 520, 100, 30)
        self.button.clicked.connect(self.recognize_plate)


    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg *.jpeg)", options=options)
        if file_name:
            self.image_label.setPixmap(QPixmap(file_name).scaled(self.image_label.size(), aspectRatioMode=1))
            self.image_path = file_name

    def recognize_plate(self):
        if self.image_path is None:
            self.result_label.setText("请选择一张图片")
            return
        locator = Locator(self.image_path)
        plates = locator.find_plate()
        segment = Segment(plates)
        plates_in_chars = segment.segment_plate()
        char_identification = CharIdentification(plates_in_chars)
        identified_plates = char_identification.identify_char()
        self.result_label.setText("识别结果: " + ", ".join(identified_plates))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlateRecognitionApp()
    window.show()
    sys.exit(app.exec_())
