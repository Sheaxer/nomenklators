import os
from typing import List, Any

import numpy as np
import cv2
from PyQt5.QtGui import QPixmap, qRgb, QImage
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QFileDialog, QMessageBox, QSlider, QLabel, \
    QHBoxLayout, QGraphicsView, QCheckBox
from PyQt5.QtCore import Qt
from scipy.signal import find_peaks
from PIL import Image
from PyQt5.QtCore import QMutex
import math
import sys
from photo_widgets import PhotoViewer, create_viewer
from matplotlib import pyplot as plt
from hough_line_transform_mapping_line_detection import handwritten_lines_detection
from image_preprocessor import binarize_image, filter_image, equalize_image, get_binary_after_gabor, preprocess_img
from slider import SliderDuo
from worker import Worker


class DialogApp(QWidget):
    def __init__(self):
        super().__init__()

        self.operationLock = QMutex()
        self.worker_thread = None
        self.worker = None

        self.debug_checker = QCheckBox("Print operation images")
        self.debug_checker.setChecked(False)
        self.debug_checker.toggled.connect(self.toggle_debug)
        self.is_debug = False

        self.export_folder = ""
        self.import_folder = ""

        self.set_export_folder_button = QPushButton("Set export folder")
        self.set_export_folder_button.clicked.connect(self.set_export_folder)

        self.image = None
        self.image_name = None
        self.image_width = None
        self.image_height = None
        self.equalized_image = None
        self.gray = None
        self.horizontal_contoured_image = None
        self.horizontal_lines_detected_image = None

        self.horizontal_line_areas = None

        self.horizontal_lines_detected_viewer = False

        self.horizontal_contours = None

        self.image_viewer = None

        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.activate_get_image)

        self.horizontal_contour_min_length = 1200
        self.horizontal_contour_min_length_slider = SliderDuo("Minimal horizontal contour length",
                                                              self.horizontal_contour_min_length, 100, 4000)
        self.horizontal_contour_min_length_slider.changed.connect(self.update_horizontal_contour_min_length)

        self.detect_horizontal_lines_button = QPushButton("Detect Horizontal Lines")
        self.detect_horizontal_lines_button.clicked.connect(self.activate_find_horizontal_lines)

        self.find_table_button = QPushButton("Find Tables")
        self.find_table_button.clicked.connect(self.activate_find_table)

        self.table = None
        self.table_viewer = None
        self.classified_image = None
        self.classified_viewer = None

        self.classification_string = None

        vb_layout = QVBoxLayout(self)

        vb_layout.addWidget(self.load_image_button)
        vb_layout.addWidget(self.horizontal_contour_min_length_slider)
        vb_layout.addWidget(self.detect_horizontal_lines_button)
        vb_layout.addWidget(self.find_table_button)
        vb_layout.addWidget(self.set_export_folder_button)

    def update_horizontal_contour_min_length(self, value):
        self.horizontal_contour_min_length = value

    def set_export_folder(self):
        if not self.operationLock.tryLock():
            return
        folder_path = QFileDialog.getExistingDirectory(self, "Set export folder", self.export_folder,
                                                       QFileDialog.ShowDirsOnly)
        if folder_path is None or not folder_path:
            self.operationLock.unlock()
            return
        self.export_folder = folder_path
        self.operationLock.unlock()

    def toggle_debug(self):
        self.is_debug = not self.is_debug

    def initiate_worker(self):
        self.worker = Worker(self.image, self.export_folder, self.is_debug, self.horizontal_contours,
                             self.horizontal_contour_min_length, self.horizontal_line_areas,
                             self.horizontal_contoured_image, self.gray,
                             self.equalized_image)

        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.finished.connect(self.worker_thread.quit)

    def activate_get_image(self):
        if not self.operationLock.tryLock():
            return
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File',
                                                   self.import_folder,
                                                   "Image files (*.jpg *.jpeg *.gif *.png)")
        if file_name is None or not file_name:
            self.operationLock.unlock()
            return

        self.image = cv2.imread(file_name)
        self.image_name = os.path.splitext(os.path.basename(file_name))[0]
        self.import_folder = os.path.split(os.path.abspath(file_name))[0]
        self.image_width = self.image.shape[1]
        self.image_height = self.image.shape[0]

        self.initiate_worker()

        self.worker_thread.started.connect(self.worker.preprocess_image)
        self.worker.send_image_image_list.connect(self.image_processed)
        self.worker_thread.start()

    def save_image(self, img, string):
        print(self.export_folder)
        cv2.imwrite(self.export_folder + "/"+ self.image_name + "-" + string + ".jpg", img)

    def save_variable(self, var, string):
        if isinstance(var, np.ndarray):
            np.savetxt(self.export_folder + "/" + self.image_name + "-" + string + ".txt", var)
        else:
            f = open(self.export_folder + "/" +self.image_name + "-" + string + ".txt", 'w')
            content = str(var)
            f.write(content)
            f.close()

    def save_table(self, table, string):
        print(self.export_folder)
        f = open(self.export_folder + "/" +  self.image_name + "-" + string + ".txt", 'w')
        for table_line in table:
            if len(table_line) > 0:
                for cell in table_line:
                    f.write(str(cell[0]) +"," + str(cell[1]) +"," + str(cell[2]) +","+ str(cell[3]) + " ")
                f.write("\n")
        f.close()


    def activate_load_image(self):
        if not self.operationLock.tryLock():
            return
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', self.import_folder,
                                                   "Image files (*.jpg *.jpeg *.gif *.png)")
        if file_name is None or not file_name:
            self.operationLock.unlock()
            return

        self.image = cv2.imread(file_name)
        self.image_name = os.path.splitext(os.path.basename(file_name))[0]

        self.image_width = self.image.shape[1]
        self.image_height = self.image.shape[0]
        self.initiate_worker()

        self.worker_thread.started.connect(self.worker.preprocess_image)
        self.worker.send_image_image_list.connect(self.image_processed)
        self.worker_thread.start()

    def image_processed(self, equalized, gray, horizontal_contours):
        self.horizontal_contours = horizontal_contours
        self.equalized_image = equalized
        self.gray = gray
        QMessageBox.about(self, "Title", "Finished Processing")
        self.image_viewer = create_viewer(self.image, "Image")

        self.table = None
        self.horizontal_line_areas = None
        self.classification_string = None
        self.horizontal_contoured_image = None

        self.operationLock.unlock()

    def activate_find_horizontal_lines(self):
        if not self.operationLock.tryLock():
            return
        if self.horizontal_contours is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_horizontal_lines)
        self.worker.detected_horizontal_lines.connect(self.finished_find_horizontal_lines)
        self.worker_thread.start()

    def finished_find_horizontal_lines(self, horizontal_line_areas, horizontal_contoured_image,
                                       horizontal_lines_detected_image):
        self.horizontal_line_areas = horizontal_line_areas
        self.horizontal_contoured_image = horizontal_contoured_image
        self.horizontal_lines_detected_image = horizontal_lines_detected_image
        self.horizontal_lines_detected_viewer = create_viewer(horizontal_lines_detected_image,
                                                              "Detected Horizontal Lines")
        self.operationLock.unlock()

    def activate_find_table(self):

        if not self.operationLock.tryLock():
            return
        if self.horizontal_contours is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        if self.horizontal_contoured_image is None:
            QMessageBox.about(self, "Title", "You have to find horizontal lines")
            self.operationLock.unlock()
            return

        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_table)
        self.worker.detected_tables.connect(self.finish_find_table)
        self.worker_thread.start()


    def finish_find_table(self,table, img, classification_string, classification_image):
        self.table = table
        self.classified_viewer = create_viewer(classification_image, "classification")
        #print(classification_string)
        self.table_viewer = create_viewer(img, "table")
        self.save_table(table,"table")
        self.save_variable(classification_string, "classification-string")
        QMessageBox.about(self, "Classification", classification_string)
        self.save_image(img,"table-image")
        self.save_image(classification_image, "classification-image")
        self.operationLock.unlock()

def exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = DialogApp()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec_())
