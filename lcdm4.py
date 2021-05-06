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

def binarize_image(img, otsu=True):
    img_binarized = None
    if otsu:
        threshold, img_binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        threshold, img_binarized = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_binarized

def equalize_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def _get_kernel(theta) -> float:
    ksize = 31
    return cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)


def filter_image(img, theta=np.pi):
    kernel = _get_kernel(theta)
    return cv2.filter2D(img, -1, kernel)


def invert(img):
    return cv2.bitwise_not(img)

def fill_horizontal_lines(img, start_row, end_row, vertical_areas, start_index, stop_index):
    for i in range(start_index, stop_index):
        j = start_row
        while j <= end_row and img[j, vertical_areas[i][1]] == 0:
            j += 1
        if img[j, vertical_areas[i][1]] == 0:
            j = start_row + int((end_row - start_row) / 2)
        k = start_row
        while k <= end_row and img[k, vertical_areas[i + 1][0]] == 0:
            k += 1
        if img[j, vertical_areas[i + 1][0]] == 0:
            k = start_row + int((end_row - start_row) / 2)
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
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888).rgbSwapped()
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                return qim.copy() if copy else qim


def preprocess_img(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # noise_removal = cv2.GaussianBlur(gray, (5,5),3)
    # noise_removal = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    equalized = equalize_image(gray)

    img_filtered = filter_image(equalized, theta=np.pi)

    _, img_vertical_binarized = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img_filtered = filter_image(equalized, theta=np.pi/2)
    _, img_horizontal_binarized = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)



    # th2, img_bin_noise = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thinned = cv2.ximgproc.thinning(img_bin_noise)
    # dilated = cv2.dilate(thinned, np.ones((1,10),np.uint8))
    # thinned2 = cv2.ximgproc.thinning(dilated)
    # thinned = cv2.dilate(thinned,np.ones((6,1),np.uint8))
    contours_vertical, hierarchy_horizontal = cv2.findContours(img_vertical_binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_horizontal, hierarchy_horizontal = cv2.findContours(img_horizontal_binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # classified_contour_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    return contours_horizontal, contours_vertical


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def get_coordiantes_after_rotation(points, angle, image_shape, mat=None, center=None):
    if center is None:
        center = (image_shape[1] / 2, image_shape[0] / 2)
    if mat is None:
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_points = []
    for point in points:
        new_point = [0, 0]

        new_point[0] = int(mat[0][0] * point[0] + mat[0][1] * point[1] + mat[0][2])
        new_point[1] = int(mat[1][0] * point[0] + mat[1][1] * point[1] + mat[1][2])
        new_point = np.array(new_point, np.uint32)
        new_points.append(new_point)
    return np.array(new_points, np.uint32)


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
    for i in range(0, data.size):
        if (0 - eps) < data[i] < eps:
            if is_area:
                if zero_vals >= minimal_gap:
                    is_area = False
                    if int(i - zero_vals - area_start) >= min_value:
                        areas.append(
                            np.array((area_start, i - zero_vals, i - zero_vals - area_start), np.uint32))
                    zero_vals = 0
                else:
                    zero_vals += 1
        else:
            if not is_area:
                is_area = True
                area_start = i
            zero_vals = 0
    if is_area:
        if int(data.shape[0] - 1 - zero_vals - area_start) >= min_value:
            areas.append(np.array((area_start, data.shape[0] - 1 - zero_vals,
                                   data.shape[0] - 1 - zero_vals - area_start), np.uint32))
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






def find_horizontal_areas(horizontal_lines_image, angle, areas_gap, min_vertical_area_value=5, min_line_value=300):
    # rotate image
    tmp_image_h_rotated = rotate(horizontal_lines_image, angle)
    # horizontal projection - trying to find horizontal lines
    horizontal_proj = horizontal_projection(tmp_image_h_rotated)
    # sepparating whole lines together
    angle_horizontal_areas = find_areas(horizontal_proj, 10, 1)
    # preparing to filter out fake lines - e.g. those that dont have sufficient length
    filtered_horizontal_areas = []
    horizontal_area_vertical_areas = []
    for area in angle_horizontal_areas:
        # do a vertical projection on the area - find the length of line segments
        vertical_proj = vertical_projection(tmp_image_h_rotated[area[0]:area[1] + 1, :])

        # trying to find lines that are not just minimal fragments left
        vertical_areas = find_areas(vertical_proj, areas_gap, min_vertical_area_value)
        # total combined length of all lines in the area
        if vertical_areas.shape[0] == 0:
            area_length = 0
        elif vertical_areas.shape[0] == 1:
            area_length = vertical_areas[0][2]
        else:
            area_length = np.sum(vertical_areas[:, 2])

        # this is to check if its not just some random noise
        if area_length > min_line_value:
            filtered_horizontal_areas.append(np.append(area, [vertical_areas[0][0], vertical_areas[-1][1]]))
            # horizontal_area_vertical_areas.append(vertical_areas)

    filtered_horizontal_areas = np.array(filtered_horizontal_areas)
    """
    filtered_horizontal_areas = np.array(filtered_horizontal_areas, np.uint32)
    filtered_horizontal_area_vertical_areas = []
    # trying to sepparate table lines from substituion leading lines
    if filtered_horizontal_areas.shape[0] > 6:
        # try to find out 3 largest lines and calculate mean and standard deviation with 1 degree of freedom
        max_values = filtered_horizontal_areas[np.argsort(filtered_horizontal_areas[:, 3])[-5:]][:,3]
        mean = np.mean(max_values)
        disp = np.std(max_values, None, None, None, 1)
        # low value
        lower = int(mean - 4 * disp)
        allowed = 0
        is_allowed = True
        filtered_horizontal_areas2 = []
        # allow short lines only if there are few of them inside longer lines - table
        # if there is bunch of short lines in succession it should mean leading lines of substitution
        for i in range(0, filtered_horizontal_areas.shape[0]):
            if filtered_horizontal_areas[i][3] > lower:
                filtered_horizontal_areas2.append(filtered_horizontal_areas[i])
                filtered_horizontal_area_vertical_areas.append(horizontal_area_vertical_areas[i])
                allowed = 0
            elif is_allowed:
                if allowed < 4:
                    allowed += 1
                    filtered_horizontal_areas2.append(filtered_horizontal_areas[i])
                    filtered_horizontal_area_vertical_areas.append(horizontal_area_vertical_areas[i])
                else:
                    is_allowed = False
                    for j in range(0, allowed):
                        filtered_horizontal_areas2.pop()
                        filtered_horizontal_area_vertical_areas.pop()
                    allowed = 0
        if allowed != 0:
            for j in range(0, allowed):
                filtered_horizontal_areas2.pop()
                filtered_horizontal_area_vertical_areas.pop()

        filtered_horizontal_areas = np.array(filtered_horizontal_areas2)
        horizontal_area_vertical_areas = filtered_horizontal_area_vertical_areas
    """
    return {"angle": angle, "areas": filtered_horizontal_areas, "image": tmp_image_h_rotated,
            "total_area": np.sum(filtered_horizontal_areas[:, 2]), "projection": horizontal_proj,
            }


def find_vertical_lines_2(vertical_lines_image, angle, horizontal_lines, horizontal_lines_vertical_areas):
    tmp_image_v_rotated = rotate(vertical_lines_image, angle)

    for i in range(0, horizontal_lines.shape[0] - 1):
        k = i + 1
        found_next_line = False
        overlapping_points = []
        start_point_1 = horizontal_lines_vertical_areas[i][0][0]
        end_point_1 = horizontal_lines_vertical_areas[i][-1][1]
        available_area = end_point_1 - start_point_1
        while k < horizontal_lines.shape[0] and available_area > 100:
            for j in range(0, horizontal_lines_vertical_areas[i].shape[0]):
                start_point = horizontal_lines_vertical_areas[i][j][0]
                end_point = horizontal_lines_vertical_areas[i][j][1]
                overlaps = []


def find_vertical_areas(vertical_lines_image, angle, areas_gap, min_area_value=1):
    tmp_image_v_rotated = rotate(vertical_lines_image, angle)
    vertical_proj = vertical_projection(tmp_image_v_rotated)
    angle_vertical_areas = find_areas(vertical_proj, 3, 1)

    filtered_vertical_areas = []
    vertical_area_horizontal_areas = []
    for area in angle_vertical_areas:
        horizontal_proj = horizontal_projection(tmp_image_v_rotated[:, area[0] - 1:area[1] + 1])
        cv2.imwrite("export/vertical-test-" + str(area[0]) + "-" + str(area[1]) + ".jpg",
                    tmp_image_v_rotated[:, area[0]: area[1] + 1])
        horizontal_areas = find_areas(horizontal_proj, 200, 15)
        if horizontal_areas.shape[0] == 0:
            area_sum = 0
        elif horizontal_areas.shape[0] == 1:
            area_sum = horizontal_areas[0][2]
        else:
            area_sum = np.sum(horizontal_areas[:, 2])

        if area_sum > min_area_value:
            filtered_vertical_areas.append(np.append(area, area_sum))
            vertical_area_horizontal_areas.append(horizontal_areas)

    filtered_vertical_areas = np.array(filtered_vertical_areas, np.uint32)

    return {"angle": angle, "areas": filtered_vertical_areas, "image": tmp_image_v_rotated,
            "total_area": np.sum(filtered_vertical_areas[:, 2]), "projection": vertical_proj,
            "horizontal_areas": vertical_area_horizontal_areas}


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

        self.operationLock = QMutex()
        self.worker_thread = None
        self.worker = None

        self.image = None
        self.image_width = 0
        self.image_height = 0
        self.image_name = None

        self.export_folder = r"export/"

        self.vertical_contours = None
        self.horizontal_contours = None

        self.horizontal_lines = None

        self.horizontal_image = None
        self.vertical_image = None
        self.horizontal_lines_image = None
        self.vertical_lines_image = None

        self.loadImageButton = QPushButton("Load Image")
        self.loadImageButton.clicked.connect(self.activate_get_image)

        self.horizontal_contour_min_length = 600
        self.horizontal_contour_min_length_slider = SliderDuo("Minimal horizontal contour length",
                                                              self.horizontal_contour_min_length, 100, 4000)
        self.horizontal_contour_min_length_slider.changed.connect(self.update_horizontal_contour_min_length)

        self.find_horizontal_lines_button = QPushButton("Find Horizontal Lines")
        self.find_horizontal_lines_button.clicked.connect(self.activate_find_horizontal_lines)

        self.vertical_contour_min_length = 20
        self.vertical_contour_min_length_slider = SliderDuo("Minimal vertical contour length",
                                                            self.vertical_contour_min_length, 10, 3000)
        self.vertical_contour_min_length_slider.changed.connect(self.update_vertical_contour_min_length)

        self.find_vertical_lines_button = QPushButton("Find Vertical Lines")
        self.find_vertical_lines_button.clicked.connect(self.activate_find_vertical_lines)

        self.find_tables_button = QPushButton("Find Tables")
        self.find_tables_button.clicked.connect(self.activate_find_tables)

        VBlayout = QVBoxLayout(self)
        # VBlayout.addWidget(self.viewer)
        # HBlayout = QHBoxLayout()
        VBlayout.addWidget(self.loadImageButton)

        VBlayout.addWidget(self.horizontal_contour_min_length_slider)
        VBlayout.addWidget(self.find_horizontal_lines_button)

        VBlayout.addWidget(self.vertical_contour_min_length_slider)
        VBlayout.addWidget(self.find_vertical_lines_button)
        VBlayout.addWidget(self.find_tables_button)
        self.table = None

        self.image_viewer = None
        self.horizontal_viewer = None
        self.vertical_viewer = None
        self.table_viewer = None
        # HBlayout.setAlignment(Qt.AlignLeft)
        # HBlayout.addWidget(self.loadImageButton)
        # VBlayout.addLayout(HBlayout)

    """
    def photoClicked(self, pos):
        if self.viewer.dragMode() == QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))
    """

    def update_horizontal_contour_min_length(self,value):
        self.horizontal_contour_min_length = value

    def update_vertical_contour_min_length(self,value):
        self.vertical_contour_min_length = value

    def initiate_worker(self):
        self.worker = Worker(self.horizontal_contours, self.horizontal_contour_min_length, self.vertical_contours,
                             self.vertical_contour_min_length, self.image, self.horizontal_lines,
                             self.horizontal_lines_image, self.vertical_lines_image)

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

    def image_processed(self, horizontal_contours, vertical_contours):
        self.horizontal_contours = horizontal_contours
        self.vertical_contours = vertical_contours
        QMessageBox.about(self, "Title", "Finished Processing")
        self.image_viewer = create_viewer(self.image,"Image")
        # self.viewer.setPhoto(QPixmap( QPixmap.fromImage(toQImage(thinned))))
        self.operationLock.unlock()
    """
    def activate_find_vertical_contours(self):
        if not self.operationLock.tryLock():
            return
        if self.thinned is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_vertical_contours)
        self.worker.processed1.connect(self.finish_finding_vertical_contours)
        self.worker_thread.start()

    def finish_finding_vertical_contours(self, image):
        self.vertical_image = image
        QMessageBox.about(self, "Title", "Vertical Contours found")
        self.vertical_contour_viewer = create_viewer(self.vertical_image, "Vertical Contours")
        self.operationLock.unlock()
    """

    def activate_find_vertical_lines(self):
        if not self.operationLock.tryLock():
            return
        if self.horizontal_contours is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        """
        if self.min_horizontal_area is None:
            QMessageBox.about(self, "Title", "You have to find horizontal lines first")
            self.operationLock.unlock()
            return
        if self.vertical_image is None:
            QMessageBox.about(self, "Title", "You have to find vertical contours first")
            self.operationLock.unlock()
            return
        """
        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_vertical_lines)
        self.worker.processed_2_images.connect(self.finished_finding_vertical_lines)
        self.worker_thread.start()

    def finished_finding_vertical_lines(self, vertical_lines_image, image):

        self.vertical_lines_image = vertical_lines_image
        self.vertical_image = image
        self.save_image(self.vertical_image, "vertical-contours-" + str(self.vertical_contour_min_length))
        self.save_image(self.image, "vertical-lines-" + str(self.vertical_contour_min_length))
        self.operationLock.unlock()
        self.vertical_viewer = create_viewer(image, "vertical contours")
        """
        self.save_variable(table,"table")
        self.table = table
        used_colors = []
        if len(table) > 0:
            img = np.copy(self.image)
            img = rotate(img, self.min_horizontal_area['angle'])
            for table_line in table:

                if len(table_line) > 0:
                    color = list(np.random.random(size=3) * 256)
                    while color in used_colors:
                        color = list(np.random.random(size=3) * 256)
                    used_colors.append(color)
                    for cell in table_line:
                        cv2.line(img, cell[0], cell[1], color, 2)
                        cv2.line(img, cell[0], cell[2], color, 2)
                        cv2.line(img, cell[1], cell[3], color, 2)
                        cv2.line(img, cell[2], cell[3], color, 2)
            img = rotate(img, 0 - self.min_horizontal_area['angle'])
            self.save_image(img,"tables-detected")
        self.operationLock.unlock()
    """
    """
    def activate_find_horizontal_contours(self):
        if not self.operationLock.tryLock():
            return
        if self.thinned is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_horizontal_contours)
        self.worker.processed1.connect(self.finished_finding_horizontal_contours)
        self.worker_thread.start()
    
    def finished_finding_horizontal_contours(self, image):
        self.horizontal_image = image
        QMessageBox.about(self, "Title", "Horizontal Contours found")
        self.horizontal_contours_viewer = create_viewer(self.horizontal_image, "Horizontal Contours")

        self.vertical_lines_viewer = None

        self.operationLock.unlock()
    """
    def activate_find_horizontal_lines(self):
        if not self.operationLock.tryLock():
            return
        if self.horizontal_contours is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_horizontal_lines)
        self.worker.processed_lines_image.connect(self.finished_finding_horizontal_lines)
        self.worker_thread.start()

    def finished_finding_horizontal_lines(self, horizontal_lines, horizontal_lines_image,image):
        self.horizontal_image = image
        self.horizontal_lines = horizontal_lines
        self.horizontal_lines_image = horizontal_lines_image
        self.save_image(image, "horizontal-lines-found")
        self.operationLock.unlock()
        self.horizontal_viewer = create_viewer(image, "Horizontal Lines Detected")
        QMessageBox.about(self, "Title", "Found Horizontal Lines")
        # print(str(self.horizontal_contour_min_length))
        # print(str(self.horizontal_contour_gap))

        """
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
        self.save_image(min_area['fixed_image'], "fixed-lines-" + print_str + print_str_2)
        # self.save_image(min_area['fixed_image_2'], "fixed-lines2-" + print_str + print_str_2)
        # self.save_variable(min_area['vertical_areas'], "horizontal-lines-vertical-areas-" + print_str + print_str_2)
        self.horizontal_lines_viewer = create_viewer(min_area['fixed_image'],
                                                     "Fixed Horizontal Lines " + print_str + print_str_2)
        """

    def activate_find_tables(self):
        if not self.operationLock.tryLock():
            return
        if self.horizontal_contours is None:
            QMessageBox.about(self, "Title", "You have to load image first")
            self.operationLock.unlock()
            return
        if self.horizontal_lines_image is None:
            QMessageBox.about(self, "Title", "You have to find horizontal lines")
            self.operationLock.unlock()
            return
        if self.vertical_lines_image is None:
            QMessageBox.about(self, "Title", "You have to find vertical lines")
            self.operationLock.unlock()
            return

        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_tables)
        self.worker.processed_list_image.connect(self.finished_finding_tables)
        self.worker_thread.start()

    def finished_finding_tables(self, table, image):
        self.table_viewer = create_viewer(image, "Higlated Table")
        self.save_image(image, "table-" + str(self.horizontal_contour_min_length) + "-" +
                        str(self.vertical_contour_min_length) + "-")
        self.save_variable(table, "table-" +str(self.horizontal_contour_min_length) + "-" +
                        str(self.vertical_contour_min_length) + "-" )
        self.table = table
        self.operationLock.unlock()


class Worker(QObject):
    processed1 = pyqtSignal(np.ndarray)
    processed2 = pyqtSignal(np.ndarray, np.ndarray)
    processed3 = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    processed4 = pyqtSignal(list, list)
    processed5 = pyqtSignal(dict)
    processed6 = pyqtSignal(list)

    processed_lines_image = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    processed_2_images = pyqtSignal(np.ndarray,np.ndarray)
    processed_list_image = pyqtSignal(list, np.ndarray)
    finished = pyqtSignal()

    def __init__(self, horizontal_contours, horizontal_contour_min_size, vertical_contours, vertical_contours_size,
                 image, horizontal_lines, horizontal_lines_image, vertical_lines_image):
        super().__init__()
        self.image = image
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.horizontal_contours = horizontal_contours
        self.horizontal_contour_min_size = horizontal_contour_min_size

        self.vertical_contours = vertical_contours
        self.vertical_contours_size = vertical_contours_size

        self.horizontal_lines = horizontal_lines
        self.vertical_lines_image = vertical_lines_image
        self.horizontal_lines_image = horizontal_lines_image

    def preprocess_image(self):
        if self.image is not None:
            horizontal_contours, vertical_contours = preprocess_img(self.image)
            self.processed4.emit(horizontal_contours, vertical_contours)
            self.finished.emit()
    """
    def find_vertical_contours(self):
        tmp_image_v = create_allowed_direction_image(self.classified_contours, self.vertical_contour_min_length,
                                                     self.vertical_contour_gap,
                                                     [7, 3],
                                                     (self.image_height, self.image_width))
        self.processed1.emit(tmp_image_v)
        self.finished.emit()
    """

    def find_vertical_lines(self):

        img = np.zeros((self.image_height,self.image_width), np.uint8)
        filtered_vertical_contours = []
        for contour in self.vertical_contours:
            if contour.shape[0] > self.vertical_contours_size:
                filtered_vertical_contours.append(contour)

        cv2.drawContours(img,filtered_vertical_contours,-1,(255,255,255))

        img2 = np.copy(self.image)
        cv2.drawContours(img2,filtered_vertical_contours,-1,(255,0,0),4)
        #first is just drawn vertical controures, second is contoures drawn contoures onto original image
        self.processed_2_images.emit(img,img2)
        self.finished.emit()

    def find_tables(self):
        table = []
        available_area_minimum = 80
        minimum_table_width = 100
        overlap_margin = 50
        for i in range(0, self.horizontal_lines.shape[0] - 1):
            table_line = []
            j = i+1
            found_next_line = False
            start_point_1 = self.horizontal_lines[i][3]
            end_point_1 = self.horizontal_lines[i][4]
            overlapping_points = []
            available_area = end_point_1 - start_point_1

            while j < self.horizontal_lines.shape[0] and available_area > available_area_minimum:
                potential_overlap_points = []

                start_point_2 = self.horizontal_lines[j][3]
                end_point_2 = self.horizontal_lines[j][4]

                overlap_start_point = start_point_1 if start_point_1 > start_point_2 else start_point_2
                overlap_end_point = end_point_1 if end_point_1 < end_point_2 else end_point_2

                if overlap_start_point < overlap_end_point:
                    potential_overlap_points.append([overlap_start_point, overlap_end_point])

                while len(potential_overlap_points) > 0:
                    overlap_start_point = potential_overlap_points[0][0]
                    overlap_end_point = potential_overlap_points[0][1]
                    potential_overlap_points.pop(0)

                    if len(overlapping_points) > 0:
                        for k in range(0, len(overlapping_points)):
                            if overlapping_points[k][1] < overlap_start_point < overlapping_points[k][2]:
                                overlap_start_point = overlapping_points[k][2]
                            if overlapping_points[k][1] < overlap_end_point <= overlapping_points[k][2]:
                                overlap_end_point = overlapping_points[k][1]
                    if overlap_start_point < overlap_end_point and (overlap_end_point - overlap_start_point) > minimum_table_width:
                        available_area = available_area - (overlap_end_point - overlap_start_point)
                        overlapping_points.append(np.array([j,overlap_start_point, overlap_end_point],np.uint32))
                        found_next_line = True
                j+=1
            if not found_next_line:
                continue

            for overlap in overlapping_points:
                tmp_overlap_1 = overlap[1] - overlap_margin
                if tmp_overlap_1 < 0:
                    tmp_overlap_1 = 0
                tmp_overlap_2 = overlap[2]
                if tmp_overlap_2 >= self.image_width:
                    tmp_overlap_2 = self.image_width - 1

                overlap = np.array([overlap[0], tmp_overlap_1, tmp_overlap_2], np.uint32)

                end_row = self.horizontal_lines[overlap[0]][1]
                tmp_image = self.vertical_lines_image[self.horizontal_lines[i][0]: end_row, tmp_overlap_1: tmp_overlap_2+1]

                vert_proj = vertical_projection(tmp_image)
                vert_areas = find_areas(vert_proj,17,1)
                if vert_areas.shape[0] == 0:
                    continue
                """
                tmp_image_highlated = np.copy(self.vertical_lines_image)
                tmp_image_highlated = cv2.cvtColor(tmp_image_highlated, cv2.COLOR_GRAY2BGR)
                for v in vert_areas:
                    cv2.line(tmp_image_highlated, (v[0]+overlap[1],self.horizontal_lines[i][0]), (v[0]+overlap[1], end_row),(255,0,0),2)
                    cv2.line(tmp_image_highlated, (v[1]+overlap[1],self.horizontal_lines[i][0]) , (v[1]+overlap[1], end_row), (0,255,0),2)
                cv2.imwrite("export/tmp-image" + str(i) + "-" + str(overlap[0]) + "-" + str(tmp_overlap_1) + "-" + str(
                    tmp_overlap_2) + ".jpg", tmp_image_highlated)
                """
                vert_areas_tmp = []


                #check if horizontally these lines cover at least half of the cell
                for area in vert_areas:
                    tmp_horizontal_proj = horizontal_projection(tmp_image[:, area[0]: area[1]+1])
                    tmp_horizontal_areas = find_areas(tmp_horizontal_proj, int((end_row - self.horizontal_lines[i][0])/2),
                                                      1)
                    if tmp_horizontal_areas.shape[0] == 0:
                        continue
                    else:
                        if tmp_horizontal_areas.shape[0] == 1:
                            horizontal_area_length = tmp_horizontal_areas[0,2]
                        else:
                            horizontal_area_length = np.sum(tmp_horizontal_areas[:,2])
                        if horizontal_area_length >= int(self.horizontal_lines[overlap[0]][0] - self.horizontal_lines[i][1] - 5):
                            vert_areas_tmp.append(area)

                vert_areas = np.array(vert_areas_tmp)

                for j in range(0, vert_areas.shape[0]-1):
                    pt_1_y = self.horizontal_lines[i][1]
                    pt_2_y = self.horizontal_lines[i][1]

                    pt_3_y = self.horizontal_lines[overlap[0]][0]
                    pt_4_y = self.horizontal_lines[overlap[0]][0]

                    #rotated_horizontal_image = rotate(self.horizontal_image,self.angle_for_vertical_lines)
                    sliced_image = self.horizontal_lines_image[self.horizontal_lines[i][0]: end_row+1,
                                   int(overlap[1] + vert_areas[j][0]): int(overlap[1] + vert_areas[j][1] + 1)]

                    sliced_horizontal_proj = horizontal_projection(sliced_image)

                    sliced_areas = find_areas(sliced_horizontal_proj,1,1)
                    if sliced_areas.shape[0] > 0:
                        pt_1_y = sliced_areas[0][1] + self.horizontal_lines[i][0]
                        pt_3_y = sliced_areas[-1][0] + self.horizontal_lines[i][0]

                    sliced_image = self.horizontal_lines_image[self.horizontal_lines[i][0]: end_row,
                                   vert_areas[j+1][0]: vert_areas[j+1][1]+1]

                    sliced_horizontal_proj = horizontal_projection(sliced_image)

                    sliced_areas = find_areas(sliced_horizontal_proj, 1, 1)

                    if sliced_areas.shape[0] > 0:
                        pt_2_y = sliced_areas[0][1] + self.horizontal_lines[i][0]
                        pt_4_y = sliced_areas[-1][0] + self.horizontal_lines[i][0]

                    if pt_1_y > self.horizontal_lines[i][1] or pt_1_y < self.horizontal_lines[i][0]:
                        pt_1_y = self.horizontal_lines[i][1]

                    if pt_2_y > self.horizontal_lines[i][1] or pt_2_y < self.horizontal_lines[i][0]:
                        pt_2_y = self.horizontal_lines[i][1]

                    if pt_3_y < self.horizontal_lines[overlap[0]][0] or pt_3_y > end_row:
                        pt_3_y = self.horizontal_lines[overlap[0]][0]

                    if pt_4_y < self.horizontal_lines[overlap[0]][0] or pt_4_y > end_row:
                        pt_4_y = self.horizontal_lines[overlap[0]][0]


                    # try to recover x coordinates

                    pt_1_x = vert_areas[j][1] + overlap[1]
                    pt_3_x = vert_areas[j][1] + overlap[1]

                    pt_2_x = vert_areas[j+1][0] + overlap[1]
                    pt_4_x = vert_areas[j+1][0] + overlap[1]

                    sliced_image = self.vertical_lines_image[self.horizontal_lines[i][0]: self.horizontal_lines[i][1],
                                   vert_areas[j][0]: vert_areas[j + 1][1] + 1]

                    sliced_vertical_projection = vertical_projection(sliced_image)
                    sliced_areas = find_areas(sliced_vertical_projection, 1,1)
                    if sliced_areas.shape[0] > 0:
                        pt_1_x = sliced_areas[0][1] + vert_areas[j][0] + overlap[1]
                        pt_2_x = sliced_areas[-1][0] + vert_areas[j][0] + overlap[1]

                    sliced_image = self.vertical_lines_image[self.horizontal_lines[overlap[0]][0]: end_row,
                                   vert_areas[j][0]: vert_areas[j + 1][1] + 1]

                    sliced_vertical_projection = vertical_projection(sliced_image)
                    sliced_areas = find_areas(sliced_vertical_projection, 1, 1)

                    if sliced_areas.shape[0] > 0:
                        pt_3_x = sliced_areas[0][1] + vert_areas[j][0] + overlap[1]
                        pt_4_x = sliced_areas[-1][0] + vert_areas[j][0] + overlap[1]

                    if pt_1_x > (overlap[1] + vert_areas[j][1]) or pt_1_x < (overlap[1] + vert_areas[j][0]):
                        pt_1_x = overlap[1] + vert_areas[j][1]

                    if pt_2_x < (overlap[1] + vert_areas[j + 1][0]) or pt_2_x > (overlap[1] + vert_areas[j + 1][1]):
                        pt_2_x = overlap[1] + vert_areas[j + 1][0]

                    if pt_3_x > (overlap[1] + vert_areas[j][1]) or pt_3_x < (overlap[1] + vert_areas[j][0]):
                        pt_3_x = overlap[1] + vert_areas[j][1]

                    if pt_4_x < (overlap[1] + vert_areas[j + 1][0]) or pt_4_x > (overlap[1] + vert_areas[j + 1][1]):
                        pt_4_x = overlap[1] + vert_areas[j + 1][0]

                    table_line.append(
                        [(pt_1_x, pt_1_y), (pt_2_x, pt_2_y), (pt_3_x, pt_3_y), (pt_4_x, pt_4_y), pt_2_x - pt_1_x,
                         pt_4_y - pt_2_y])

            table.append(table_line)

        used_colors = []
        if len(table) > 0:
            img = np.copy(self.image)
            #img = rotate(img, self.min_horizontal_area['angle'])
            for table_line in table:

                if len(table_line) > 0:
                    color = list(np.random.random(size=3) * 256)
                    while color in used_colors:
                        color = list(np.random.random(size=3) * 256)
                    used_colors.append(color)
                    for cell in table_line:
                        cv2.line(img, cell[0], cell[1], color, 2)
                        cv2.line(img, cell[0], cell[2], color, 2)
                        cv2.line(img, cell[1], cell[3], color, 2)
                        cv2.line(img, cell[2], cell[3], color, 2)

        self.processed_list_image.emit(table,img)
        self.finished.emit()
    """
    def find_horizontal_contours(self):
        tmp_image_h = create_allowed_direction_image(self.classified_contours, self.horizontal_contour_min_length,
                                                     self.horizontal_contour_gap,
                                                     [1, 5],
                                                     (self.image_height, self.image_width))
        self.processed1.emit(tmp_image_h)
        self.finished.emit()
    """

    def find_horizontal_lines(self):
        """
        tmp_image_h = self.horizontal_image
        min_area = find_horizontal_areas(tmp_image_h, -2.0, self.horizontal_line_segments_gap,
                                         self.horizontal_line_segment_size, self.horizontal_lines_min_size)
        # hist, _ = np.histogram((tmp_image_h / 255).ravel(), 256,[0,255] )
        # np.savetxt("export/hist.txt",hist)
        # debug_save_finding_horizontal_areas("img/", min_area)
        for angle in np.arange(-1.5, 2.5, 0.5):
            tmp_area = find_horizontal_areas(tmp_image_h, angle, self.horizontal_line_segments_gap,
                                             self.horizontal_line_segment_size, self.horizontal_lines_min_size)
            # debug_save_finding_horizontal_areas("img/", tmp_area)
            if tmp_area['total_area'] < min_area['total_area']:
                min_area = tmp_area

        # for area in min_area['areas']:
        horizontal_area_vertical_areas = []
        img2 = np.zeros_like(min_area['image'])


        change_margin = 150

        min = min_area['areas'][np.argsort(min_area['areas'][:,3])][0,3]
        max = min_area['areas'][np.argsort(min_area['areas'][:,4])][-1,4]
        for area in min_area['areas']:
            if area[3] != min and area[3] < (min+change_margin):
                area[3] = min
            if area[4] != max and area[4] > (max - change_margin):
                area[4] = max

        for area in min_area['areas']:
            row = area[0] + int((area[1] - area[0]) / 2)
            cv2.line(img2, (area[3], row), (area[4], row), (255, 255, 255), 5)

        min_area['fixed_image'] = img2
        # cv2.imwrite("export/fixed_lines.jpg",img2)
        """
        img = np.zeros((self.image_height,self.image_width), np.uint8)
        filtered_horizontal_contours = []
        for contour in self.horizontal_contours:
            if contour.shape[0] > self.horizontal_contour_min_size:
                filtered_horizontal_contours.append(contour)

        cv2.drawContours(img, filtered_horizontal_contours, -1, (255, 255, 255))

        horizontal_proj = horizontal_projection(img)
        horizontal_areas = find_areas(horizontal_proj, 10, 1)
        horizontal_line_areas = []
        for area in horizontal_areas:
            sliced_vertical_img = img[area[0]:area[1], :]
            vert_proj = vertical_projection(sliced_vertical_img)
            vert_areas = find_areas(vert_proj,2,1)
            horizontal_line_areas.append(np.append(area, [vert_areas[0][0], vert_areas[-1][1]]))
        horizontal_line_areas = np.array(horizontal_line_areas)

        start_point = horizontal_line_areas[np.argsort(horizontal_line_areas[:,3])][0,3]
        end_point = horizontal_line_areas[np.argsort(horizontal_line_areas[:,4])][-1,4]

        for area in horizontal_line_areas:
            if area[3] < (start_point + 100):
                area[3] = start_point
            if area[4] < (end_point - 100):
                area[4] = end_point

        img2 = np.copy(self.image)
        for area in horizontal_line_areas:
            cv2.line(img2,(area[3], area[0]), (area[4], area[0]), (255,0,0),4)
            cv2.line(img2,(area[3],area[1]), (area[4], area[1]), (0,255,0),4)

        self.processed_lines_image.emit(horizontal_line_areas, img, img2)
        #self.processed5.emit(min_area)
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
