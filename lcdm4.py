import os
from typing import List, Any

import numpy as np
import cv2
from PyQt5.QtGui import QPixmap, qRgb, QImage
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QFileDialog, QMessageBox, QSlider, QLabel, \
    QHBoxLayout, QGraphicsView
from PyQt5.QtCore import Qt
from scipy.signal import find_peaks
from PIL import Image
from PyQt5.QtCore import QMutex
import math
import sys
from photo import PhotoViewer
from matplotlib import pyplot as plt

gray_color_table = [qRgb(i, i, i) for i in range(256)]


def fill_horizontal_lines(img, start_row, vertical_areas, start_index, stop_index):
    for i in range(start_index, stop_index):
        j = start_row
        while img[j, vertical_areas[i][1]] == 0:
            j += 1
        k = start_row
        while img[k, vertical_areas[i + 1][0]] == 0:
            k += 1
        cv2.line(img, (vertical_areas[i][1], j), (vertical_areas[i + 1][0], k), (255, 255, 255), 1)
    return img


def fill_vertical_lines(img, start_column, vertical_areas, start_index, stop_index):
    for i in range(start_index, stop_index):
        j = start_column
        while img[j, vertical_areas[i][1]] == 0:
            j += 1
        k = start_column
        while img[k, vertical_areas[i + 1][0]] == 0:
            k += 1
        cv2.line(img, (j, vertical_areas[i][1]), (k, vertical_areas[i + 1][0]), (255, 255, 255), 1)
    return img


def create_viewer(image, title):
    a = PhotoWindow()
    a.set_photo(image)
    a.setWindowTitle(title)
    a.setGeometry(500, 300, 800, 600)
    a.show()
    return a


def toQImage(im, copy=False):
    if im is None:
        return QImage()
    im = np.require(im, np.uint8, 'C')
    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                return qim.copy() if copy else qim


def preprocess_img(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # noise_removal = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    th2, img_bin_noise = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thinned = cv2.ximgproc.thinning(img_bin_noise)
    # thinned = cv2.dilate(thinned,np.ones((6,1),np.uint8))
    contours, hierarchy = cv2.findContours(thinned, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # classified_contour_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    classified_contours = contour_classification(contours)
    return classified_contours, thinned, img_bin_noise, gray, gray


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def vertical_projection(img, flip=False):
    if flip:
        image = ~img
    else:
        image = np.copy(img)
    image = image / 255
    data = np.sum(image, 0, np.uint32)
    return data


def horizontal_projection(img, flip=False):
    if flip:
        image = ~img
    else:
        image = np.copy(img)
    image = image / 255
    data = np.sum(image, 1, np.uint32)
    return data


def horizontal_projection_graph(data, shape):
    (h, w) = shape[:2]
    graph = np.zeros((h, 500), np.uint8)
    m = np.amax(data)
    for i in range(0, graph.shape[0]):
        for j in range(0, graph.shape[1]):
            graph[i][j] = 255

    for i in range(0, data.shape[0]):
        cv2.line(graph, (0, i), (int((data[i] / m) * 500), i), (0, 0, 0), 1)
    return graph


def vertical_projection_graph(data, shape):
    (h, w) = shape[:2]
    graph = np.zeros((500, w), np.uint8)
    m = np.amax(data)
    for i in range(0, graph.shape[0]):
        for j in range(0, graph.shape[1]):
            graph[i][j] = 255
    if m == 0.0:
        return graph, data
    for i in range(0, data.shape[0]):
        cv2.line(graph, (i, 500 - int((data[i] / m) * 500)), (i, 500), (0, 0, 0), 1)
    return graph


def find_areas(data, minimal_gap=2, min_value=1):
    is_area = False
    eps = 0.01
    zero_vals = 0
    areas = []
    area_start = 0
    area_sum = 0
    for i in range(0, data.size):
        if (0 - eps) < data[i] < eps:
            if is_area:
                if zero_vals >= minimal_gap:
                    is_area = False
                    if int(area_sum) >= min_value:
                        areas.append(
                            np.array((area_start, i - zero_vals, i - zero_vals - area_start, int(area_sum)), np.uint32))
                    zero_vals = 0
                    area_sum = 0
                else:
                    zero_vals += 1
        else:
            if not is_area:
                is_area = True
                area_start = i
            zero_vals = 0
            area_sum += data[i]
    if is_area:
        if int(area_sum) >= min_value:
            areas.append(np.array((area_start, data.shape[0] - 1 - zero_vals,
                                   data.shape[0] - 1 - zero_vals - area_start, int(area_sum)), np.uint32))
    areas = np.array(areas)
    return areas


class ContourFragment:
    def __init__(self, classification, start_point, end_point):
        self.classification = classification
        self.start_point = start_point
        self.end_point = end_point


class ClassifiedLine:
    def __init__(self, classification, fragments=None):
        if fragments is None:
            fragments = []
        self.classification = classification
        self.fragments = fragments

    def add_fragment(self, fragment):
        self.fragments.append(fragment)


def contour_classification(contours):
    classification_matrix = [[6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8],
                             [6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8],
                             [6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8],
                             [5, 5, 6, 6, 7, 7, 7, 8, 8, 1, 1],
                             [5, 5, 5, 5, 0, 0, 0, 1, 1, 1, 1],
                             [5, 5, 5, 5, 0, 0, 0, 1, 1, 1, 1],
                             [5, 5, 5, 5, 0, 0, 0, 1, 1, 1, 1],
                             [5, 5, 4, 4, 3, 3, 3, 2, 2, 1, 1],
                             [4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2],
                             [4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2],
                             [4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2]]

    classified_contours = []
    for contour in contours:
        classified_fragments = []
        i = 0
        j = 5
        while j < contour.shape[0]:
            diff_x = round(contour[j][0][0] - contour[i][0][0]) + 5
            diff_y = round(contour[j][0][1] - contour[i][0][1]) + 5
            classification = classification_matrix[diff_y][diff_x]
            classfied_fragment = ContourFragment(classification,
                                                 (contour[i][0][0], contour[i][0][1]),
                                                 (contour[j][0][0], contour[j][0][1]))
            classified_fragments.append(classfied_fragment)
            i = j
            j += 5
        j = contour.shape[0] - 1

        diff_x = round(contour[j][0][0] - contour[i][0][0]) + 5
        diff_y = round(contour[j][0][1] - contour[i][0][1]) + 5
        classification = classification_matrix[diff_y][diff_x]
        classfied_fragment = ContourFragment(classification,
                                             (contour[i][0][0], contour[i][0][1]),
                                             (contour[j][0][0], contour[j][0][1]))
        classified_fragments.append(classfied_fragment)

        classified_contours.append(classified_fragments)

    return classified_contours


def find_line_fragments_2(classified_contours, min_fragments=3, max_gap=2):
    lines = []
    for classified_contour in classified_contours:
        g = 0
        tmp_line = []
        is_line = False
        current_classification = None
        i = 0
        while i < len(classified_contour):
            if is_line:
                if (classified_contour[i].classification == current_classification) \
                        or (classified_contour[i].classification == 0):
                    tmp_line.append(classified_contour[i])
                    i += 1
                elif g < max_gap:
                    g += 1
                    tmp_line.append(classified_contour[i])
                    i += 1
                else:
                    for k in range(0, g):
                        tmp_line.pop()
                    is_line = False
                    i -= g
                    g = 0
                    if len(tmp_line) >= min_fragments:
                        lines.append(ClassifiedLine(current_classification, tmp_line))
                    tmp_line = []
                    current_classification = None
            else:
                if classified_contour[i].classification == 0:
                    i += 1
                else:
                    is_line = True
                    current_classification = classified_contour[i].classification
                    tmp_line.append(classified_contour[i])
                    i += 1
        if len(tmp_line) >= min_fragments:
            if current_classification is None:
                current_classification = 0
            lines.append(ClassifiedLine(current_classification, tmp_line))
    return lines


def find_line_fragments(classified_contours, allowed_directions, min_fragments=3, max_gap=2):
    lines = []
    for classified_contour in classified_contours:
        g = 0
        tmp_line = []
        is_line = False
        for i in range(0, len(classified_contour)):
            if (classified_contour[i].classification in allowed_directions) or (
                    classified_contour[i].classification == 0):
                tmp_line.append(classified_contour[i])
                is_line = True
            else:
                if is_line:
                    if g <= max_gap:
                        tmp_line.append(classified_contour[i])
                        g += 1
                    else:
                        is_line = False
                        for k in range(0, g + 1):
                            tmp_line.pop()
                            g = 0
                        if len(tmp_line) >= min_fragments:
                            lines.append(tmp_line)
                        tmp_line = []
    return lines


def create_allowed_direction_image(classified_contours, minimal_size, max_gap, allowed_directions, shape):
    line_fragments = find_line_fragments_2(classified_contours, minimal_size, max_gap)
    tmp_image = np.zeros(shape, np.uint8)
    for line in line_fragments:
        if line.classification in allowed_directions:
            for fragment in line.fragments:
                cv2.line(tmp_image, fragment.start_point, fragment.end_point, (255, 255, 255), 1)
    return tmp_image


def find_horizontal_areas(horizontal_lines_image, angle, areas_gap, min_area_value=1):
    tmp_image_h_rotated = rotate(horizontal_lines_image, angle)
    horizontal_proj = horizontal_projection(tmp_image_h_rotated)
    angle_horizontal_areas = find_areas(horizontal_proj, areas_gap, min_area_value)
    return {"angle": angle, "areas": angle_horizontal_areas, "image": tmp_image_h_rotated,
            "total_area": np.sum(angle_horizontal_areas[:, 2]), "projection": horizontal_proj}


def find_vertical_areas(vertical_lines_image, angle, areas_gap, min_area_value=1):
    tmp_image_v_rotated = rotate(vertical_lines_image, angle)
    vertical_proj = vertical_projection(tmp_image_v_rotated)
    angle_vertical_areas = find_areas(vertical_proj, areas_gap, min_area_value)
    return {"angle": angle, "areas": angle_vertical_areas, "image": tmp_image_v_rotated,
            "total_area": np.sum(angle_vertical_areas[:, 2]), "projection": vertical_proj}


class SliderDuo(QWidget):
    changed = pyqtSignal(int)

    def __init__(self, text, default_value, min_value, max_value):
        super().__init__()
        self.textLabel = QLabel(text)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_value, max_value)
        self.slider.setValue(default_value)
        self.value = default_value
        self.label = QLabel(str(self.value))
        self.label.setStyleSheet('QLabel { background: #007AA5; border-radius: 3px;}')
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label.setMinimumWidth(80)
        self.slider.valueChanged.connect(self.change_value)
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(self.textLabel)
        hbox.addWidget(self.slider)
        hbox.addSpacing(15)
        hbox.addWidget(self.label)
        hbox.addStretch()

        self.setLayout(hbox)

    def change_value(self, value):
        self.label.setText(str(value))
        self.value = int(value)
        self.changed.emit(int(value))


class DialogApp(QWidget):
    def __init__(self):
        super().__init__()

        # self.viewer = PhotoViewer(self)
        # self.viewer.photoClicked.connect(self.photoClicked)
        self.thinned_viewer = None
        self.horizontal_viewer = None
        self.vertical_viewer = None
        self.restored_horizontal_viewer = None
        self.restored_vertical_viewer = None

        self.classified_contours = None
        self.thinned = None
        self.binarized = None
        self.denoised = None
        self.gray = None

        self.horizontal_image = None
        self.vertical_image = None

        self.min_vertical_area = None
        self.min_horizontal_area = None

        self.horizontal_lines = None
        self.vertical_lines = None

        self.horizontal_lines_vertical_areas = None
        self.vertical_lines_horizontal_areas = None

        self.operationLock = QMutex()
        self.worker_thread = None
        self.worker = None

        self.image = None
        self.image_width = 0
        self.image_height = 0
        self.image_name = None

        self.horizontal_contour_min_length = 12
        self.vertical_contour_min_length = 6

        self.horizontal_contour_length_slider = SliderDuo("Minimal horizontal contour length",
                                                          self.horizontal_contour_min_length, 1, 60)
        self.horizontal_contour_length_slider.changed.connect(self.update_horizontal_contour_min_length)

        self.vertical_contour_length_slider = SliderDuo("Minimal vertical contour length",
                                                        self.vertical_contour_min_length, 1, 60)
        self.vertical_contour_length_slider.changed.connect(self.update_vertical_contour_min_length)

        self.export_folder = r"export/"

        self.loadImageButton = QPushButton("Load Image")
        self.loadImageButton.clicked.connect(self.activate_get_image)

        self.find_horizontal_lines_button = QPushButton("Find Horizontal Contours")
        self.find_horizontal_lines_button.clicked.connect(self.activate_find_horizontal_lines)
        self.find_vertical_lines_button = QPushButton("Find Vertical Contours")
        self.find_vertical_lines_button.clicked.connect(self.activate_find_vertical_lines)

        self.horizontal_contour_gap = 4
        self.vertical_contour_gap = 4

        self.horizontal_contour_gap_slider = SliderDuo("Horizontal contour gap", self.horizontal_contour_gap, 1, 20)
        self.horizontal_contour_gap_slider.changed.connect(self.update_horizontal_contour_gap)

        self.vertical_contour_gap_slider = SliderDuo("Vertical contour gap", self.vertical_contour_gap, 1, 20)
        self.vertical_contour_gap_slider.changed.connect(self.update_vertical_contour_gap)

        self.horizontal_lines_merge_size = 150
        self.horizontal_lines_merge_size_slider = SliderDuo("Horizontal lines merge length",
                                                            self.horizontal_lines_merge_size, 40, 500)
        self.horizontal_lines_merge_size_slider.changed.connect(self.update_horizontal_lines_merge_size)

        self.vertical_lines_merge_size = 50
        self.vertical_lines_merge_size_slider = SliderDuo("Vertical lines merge length",
                                                          self.vertical_lines_merge_size, 40, 500)
        self.vertical_lines_merge_size_slider.changed.connect(self.update_vertical_lines_merge_size)

        self.horizontal_lines_min_size = 10
        self.horizontal_lines_min_size_slider = SliderDuo("Horizontal lines minimal size",
                                                          self.horizontal_lines_min_size, 1, 60)
        self.horizontal_lines_min_size_slider.changed.connect(self.update_horizontal_lines_min_size)

        self.vertical_lines_min_size = 5
        self.vertical_lines_min_size_slider = SliderDuo("Vertical lines minimal size",
                                                        self.vertical_lines_min_size, 1, 60)
        self.vertical_lines_min_size_slider.changed.connect(self.update_vertical_lines_min_size)

        VBlayout = QVBoxLayout(self)
        # VBlayout.addWidget(self.viewer)
        # HBlayout = QHBoxLayout()
        VBlayout.addWidget(self.loadImageButton)

        VBlayout.addWidget(self.horizontal_contour_length_slider)
        VBlayout.addWidget(self.horizontal_contour_gap_slider)
        VBlayout.addWidget(self.horizontal_lines_min_size_slider)
        VBlayout.addWidget(self.horizontal_lines_merge_size_slider)
        VBlayout.addWidget(self.find_horizontal_lines_button)

        VBlayout.addWidget(self.vertical_contour_length_slider)
        VBlayout.addWidget(self.vertical_contour_gap_slider)
        VBlayout.addWidget(self.vertical_lines_min_size_slider)
        VBlayout.addWidget(self.vertical_lines_merge_size_slider)
        VBlayout.addWidget(self.find_vertical_lines_button)

        # HBlayout.setAlignment(Qt.AlignLeft)
        # HBlayout.addWidget(self.loadImageButton)
        # VBlayout.addLayout(HBlayout)

    """
    def photoClicked(self, pos):
        if self.viewer.dragMode() == QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))
    """

    def update_horizontal_lines_merge_size(self, value):
        self.horizontal_lines_merge_size = value

    def update_vertical_lines_merge_size(self, value):
        self.vertical_lines_merge_size = value

    def update_horizontal_lines_min_size(self, value):
        self.horizontal_lines_min_size = value

    def update_horizontal_contour_min_length(self, value):
        self.horizontal_contour_min_length = value

    def update_vertical_lines_min_size(self, value):
        self.vertical_lines_min_size = value

    def update_horizontal_contour_gap(self, value):
        self.horizontal_contour_gap = value

    def update_vertical_contour_min_length(self, value):
        self.vertical_contour_min_length = value

    def update_vertical_contour_gap(self, value):
        self.vertical_contour_gap = value

    def initiate_worker(self):
        self.worker = Worker(self.image, self.classified_contours,
                             self.horizontal_contour_min_length, self.horizontal_contour_gap,
                             self.horizontal_lines_min_size, self.horizontal_lines_merge_size,
                             self.vertical_contour_min_length, self.vertical_contour_gap,
                             self.vertical_lines_min_size, self.vertical_lines_merge_size)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.finished.connect(self.worker_thread.quit)

    def activate_get_image(self):
        if not self.operationLock.tryLock():
            return
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File',
                                                   r"C:\school\Herr Eugen Antal_2020-09-23_220056\Herr Eugen Antal",
                                                   "Image files (*.jpg *.jpeg *.gif *.png)")
        if file_name is None or not file_name:
            self.operationLock.unlock()
            return

        self.image = cv2.imread(file_name)
        self.image_name = os.path.splitext(os.path.basename(file_name))[0]
        # self.modified_image = np.copy(self.image)
        # self.initiate_worker()
        # self.worker_thread.started.connect(self.worker.preprocess)
        # self.worker.processed1.connect(self.save_modified_image)
        # self.worker_thread.start()
        QMessageBox.about(self, "Title", "Image loaded")
        self.image_width = self.image.shape[1]
        self.image_height = self.image.shape[0]
        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.preprocess_image)
        self.worker.processed4.connect(self.image_processed)
        self.worker_thread.start()

    def save_image(self, img, string):
        cv2.imwrite(self.export_folder + self.image_name + "-" + string + ".jpg", img)

    def save_variable(self, var, string):
        if isinstance(var, np.ndarray):
            np.savetxt(self.export_folder + self.image_name + "-" + string + ".txt", var)
        else:
            f = open(self.export_folder + self.image_name + "-" + string + ".txt", 'w')
            content = str(var)
            f.write(content)
            f.close()

    def image_processed(self, classified_contours, thinned, binarized, denoised, gray):
        self.classified_contours = classified_contours
        self.thinned = thinned
        self.binarized = binarized
        self.denoised = denoised
        self.gray = gray
        QMessageBox.about(self, "Title", "Finished Processing")
        # self.viewer.setPhoto(QPixmap( QPixmap.fromImage(toQImage(thinned))))
        self.save_image(thinned, "thinned")
        self.save_image(binarized, "binarized")
        self.save_image(denoised, "denoised")
        self.save_image(gray, "gray")
        self.thinned_viewer = create_viewer(thinned, "thinned")
        self.operationLock.unlock()

    def activate_find_vertical_lines(self):
        if not self.operationLock.tryLock():
            return
        if self.thinned is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_vertical_lines)
        self.worker.processed5.connect(self.finished_finding_vertical_lines)
        self.worker_thread.start()

    def finished_finding_vertical_lines(self, min_area, image):
        self.min_vertical_area = min_area
        self.vertical_image = image
        QMessageBox.about(self, "Title", "Found Vertical Lines")
        print_str = str(self.vertical_contour_min_length) + "-" + \
                    str(self.vertical_contour_gap) + "_" + str(min_area['angle'])
        self.save_image(image,
                        "vertical-image" + str(self.vertical_contour_min_length) + "-" + str(self.vertical_contour_gap))
        self.save_image(min_area['image'], "vertical-image" + print_str)

        self.save_variable(min_area['areas'], "vertical-line-areas-" + print_str)
        self.vertical_viewer = create_viewer(min_area['image'], "Vertical-Lines - " + print_str)
        self.operationLock.unlock()

    def activate_find_horizontal_lines(self):
        if not self.operationLock.tryLock():
            return
        if self.thinned is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_horizontal_lines)
        self.worker.processed5.connect(self.finished_finding_horizontal_lines)
        self.worker_thread.start()

    def finished_finding_horizontal_lines(self, min_area, image):
        self.min_horizontal_area = min_area
        self.horizontal_image = image
        QMessageBox.about(self, "Title", "Found Horizontal Lines")
        # print(str(self.horizontal_contour_min_length))
        # print(str(self.horizontal_contour_gap))

        print_str = str(self.horizontal_contour_min_length) + "-" + \
                    str(self.horizontal_contour_gap) + "_" + str(min_area['angle'])
        print_str_2 = "-" + str(self.horizontal_lines_min_size) + "-" + str(self.horizontal_lines_merge_size)

        self.save_image(image, "horizontal-image" + str(self.horizontal_contour_min_length) + "-" +
                        str(self.horizontal_contour_gap))

        self.save_image(min_area['image'], "horizontal-image-" + print_str)

        self.save_variable(min_area['areas'], "horizontal-line-areas-" + print_str)

        self.horizontal_viewer = create_viewer(min_area['image'], "Horizontal-Lines -" + print_str)
        graph = horizontal_projection_graph(min_area['projection'], self.horizontal_image.shape)
        graph = cv2.cvtColor(graph, cv2.COLOR_GRAY2BGR)
        second_image = cv2.cvtColor(min_area['image'], cv2.COLOR_GRAY2BGR)
        for area in min_area['areas']:
            cv2.line(graph, (0, int(area[0])), (graph.shape[1] - 1, int(area[0])), (255, 0, 0), 2)
            cv2.line(second_image, (0, int(area[0])), (second_image.shape[1] - 1, int(area[0])), (255, 0, 0), 2)
            cv2.line(graph, (0, int(area[1])), (graph.shape[1] - 1, int(area[1])), (0, 255, 0), 2)
            cv2.line(second_image, (0, int(area[1])), (second_image.shape[1] - 1, int(area[1])), (0, 255, 0), 2)

        self.save_image(graph, "Horizontal-Areas-" + print_str)
        self.save_image(second_image, "highlated-lines-" + print_str)
        self.save_image(min_area['fixed-image'], "fixed-lines-" + print_str + print_str_2)
        self.save_variable(min_area['vertical-areas'], "horizontal-lines-vertical-areas-" + print_str + print_str_2)
        # create_viewer(min_area['fixed-image'], "Fixed Horizontal Lines " + print_str + print_str_2)
        self.operationLock.unlock()


class Worker(QObject):
    processed1 = pyqtSignal(np.ndarray)
    processed2 = pyqtSignal(np.ndarray, np.ndarray)
    processed3 = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    processed4 = pyqtSignal(list, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    processed5 = pyqtSignal(dict, np.ndarray)
    finished = pyqtSignal()

    def __init__(self, image, classified_contours,
                 horizontal_contour_min_length, horizontal_contour_gap,
                 horizontal_lines_min_size, horizontal_lines_merge_size,
                 vertical_contour_min_length, vertical_contour_gap,
                 vertical_lines_min_size, vertical_lines_merge_size):
        super().__init__()
        self.image = image
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.classified_contours = classified_contours
        self.horizontal_contour_min_length = horizontal_contour_min_length
        self.horizontal_contour_gap = horizontal_contour_gap
        self.horizontal_lines_min_size = horizontal_lines_min_size
        self.horizontal_lines_merge_size = horizontal_lines_merge_size

        self.vertical_contour_min_length = vertical_contour_min_length
        self.vertical_contour_gap = vertical_contour_gap
        self.vertical_lines_min_size = vertical_lines_min_size
        self.vertical_lines_merge_size = vertical_lines_merge_size

    def preprocess_image(self):
        if self.image is not None:
            classified_contours, thinned, binarized, denoised, gray = preprocess_img(self.image)
            self.processed4.emit(classified_contours, thinned, binarized, denoised, gray)
            self.finished.emit()

    def find_vertical_lines(self):
        tmp_image_v = create_allowed_direction_image(self.classified_contours, self.vertical_contour_min_length,
                                                     self.vertical_contour_gap,
                                                     [7, 3],
                                                     (self.image_height, self.image_width))
        min_area = find_vertical_areas(tmp_image_v, -2.0, 1)

        # debug_save_finding_horizontal_areas("img/", min_area)
        for angle in np.arange(-1.5, 2.5, 0.5):
            tmp_area = find_vertical_areas(tmp_image_v, angle, 2, 3)
            # debug_save_finding_horizontal_areas("img/", tmp_area)
            if tmp_area['total_area'] < min_area['total_area']:
                min_area = tmp_area

        self.processed5.emit(min_area, tmp_image_v)
        self.finished.emit()

    def find_horizontal_lines(self):
        tmp_image_h = create_allowed_direction_image(self.classified_contours, self.horizontal_contour_min_length,
                                                     self.horizontal_contour_gap,
                                                     [1, 5, 2, 8],
                                                     (self.image_height, self.image_width))
        min_area = find_horizontal_areas(tmp_image_h, -2.0, 10, self.horizontal_lines_min_size)
        # hist, _ = np.histogram((tmp_image_h / 255).ravel(), 256,[0,255] )
        # np.savetxt("export/hist.txt",hist)
        # debug_save_finding_horizontal_areas("img/", min_area)
        for angle in np.arange(-1.5, 2.5, 0.5):
            tmp_area = find_horizontal_areas(tmp_image_h, angle, 10, self.horizontal_lines_min_size)
            # debug_save_finding_horizontal_areas("img/", tmp_area)
            if tmp_area['total_area'] < min_area['total_area']:
                min_area = tmp_area

        # for area in min_area['areas']:
        horizontal_lines_vertical_areas = []
        img2 = np.copy(min_area['image'])
        for area in min_area['areas']:

            vert_proj = vertical_projection(min_area['image'][area[0]:area[1], :])
            vert_areas = find_areas(vert_proj, 1, 1)
            merged_vert_areas = []
            print("area-" + str(area[0]) + " - " + str(area[1]))
            start_index = 0
            for i in range(1, vert_areas.shape[0] - 1):
                print(str(vert_areas[i + 1][0] - vert_areas[i][1]))
                gap = vert_areas[i + 1][0] - vert_areas[i][1]
                if gap >= self.horizontal_lines_merge_size:
                    if start_index != i:
                        merged_vert_areas.append(np.array([vert_areas[start_index][0], vert_areas[i][1],
                                                           vert_areas[i][1] - vert_areas[start_index][0],
                                                           np.sum(vert_areas[start_index:i, 3])]))
                        """
                        cv2.line(img2, (vert_areas[start_index][1],area[0]),
                                 (vert_areas[i][0], area[0]), (255,255,255),1)
                        """
                        img2 = fill_horizontal_lines(img2, area[0], vert_areas, start_index, i)
                    else:
                        merged_vert_areas.append(vert_areas[i])
                    start_index = i + 1

            if start_index != vert_areas.shape[0] - 1:
                merged_vert_areas.append(np.array([vert_areas[start_index][0], vert_areas[-1][1],
                                                   vert_areas[-1][1] - vert_areas[start_index][0],
                                                   np.sum(vert_areas[start_index:-1, 3])]))
                """
                cv2.line(img2, (vert_areas[start_index][1], area[0]),
                         (vert_areas[-1][0], area[0]), (255, 255, 255), 1)
                 """
                img2 = fill_horizontal_lines(img2, area[0], vert_areas, start_index, vert_areas.shape[0] - 1)
            else:
                merged_vert_areas.append(vert_areas[-1])

            horizontal_lines_vertical_areas.append(np.array(merged_vert_areas, np.uint32))
        min_area['vertical-areas'] = horizontal_lines_vertical_areas
        min_area['fixed-image'] = img2
        # cv2.imwrite("export/fixed_lines.jpg",img2)
        self.processed5.emit(min_area, tmp_image_h)
        self.finished.emit()


class PhotoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = PhotoViewer(self)
        VBlayout = QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)

    def set_photo(self, image):
        self.viewer.setPhoto(QPixmap(QPixmap.fromImage(toQImage(image))))


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
