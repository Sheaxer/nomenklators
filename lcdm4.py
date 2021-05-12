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
from louloudis import get_number_of_lines

gray_color_table = [qRgb(i, i, i) for i in range(256)]


def binarize_image(img, otsu=True):
    img_binarized = None
    if otsu:
        threshold, img_binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        threshold, img_binarized = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return img_binarized


def equalize_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
    """
    img_filtered = filter_image(equalized, theta=np.pi)
    
    _, img_vertical_binarized = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    img_filtered = filter_image(equalized, theta=np.pi / 2)
    _, img_horizontal_binarized = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # th2, img_bin_noise = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thinned = cv2.ximgproc.thinning(img_bin_noise)
    # dilated = cv2.dilate(thinned, np.ones((1,10),np.uint8))
    # thinned2 = cv2.ximgproc.thinning(dilated)
    # thinned = cv2.dilate(thinned,np.ones((6,1),np.uint8))
  
    contours_vertical, hierarchy_horizontal = cv2.findContours(img_vertical_binarized, cv2.RETR_LIST,
                                                               cv2.CHAIN_APPROX_NONE)
   
    contours_horizontal, hierarchy_horizontal = cv2.findContours(img_horizontal_binarized, cv2.RETR_LIST,
                                                                 cv2.CHAIN_APPROX_NONE)
    """
    # classified_contour_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    # return equalized, contours_horizontal
    return equalized, gray


def get_vertical_contours(equalized_image):
    img_filtered = filter_image(equalized_image, theta=np.pi)

    _, img_vertical_binarized = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_vertical, hierarchy_horizontal = cv2.findContours(img_vertical_binarized, cv2.RETR_LIST,
                                                               cv2.CHAIN_APPROX_NONE)
    return contours_vertical


def get_horizontal_contours(equalized_image):
    img_filtered = filter_image(equalized_image, theta=np.pi / 2)
    _, img_horizontal_binarized = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_horizontal, hierarchy_horizontal = cv2.findContours(img_horizontal_binarized, cv2.RETR_LIST,
                                                                 cv2.CHAIN_APPROX_NONE)
    return contours_horizontal


def get_binary_after_gabor(gray):
    img_filtered_vertical = filter_image(gray.copy(), theta=np.pi)
    img_vertical_binarized = binarize_image(img_filtered_vertical)

    cv2.imwrite("export/bin_vert.jpg",img_vertical_binarized)

    img_filtered_horizontal = filter_image(gray.copy(), theta= np.pi / 2)
    img_horizontal_binarized = binarize_image(img_filtered_horizontal)

    cv2.imwrite("export/bin_horizont.jpg",img_horizontal_binarized)

    img = cv2.bitwise_or(img_vertical_binarized, img_horizontal_binarized)

    return img


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

        self.gray = None

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
        """
        self.vertical_contour_min_length = 20
        self.vertical_contour_min_length_slider = SliderDuo("Minimal vertical contour length",
                                                            self.vertical_contour_min_length, 10, 3000)
        self.vertical_contour_min_length_slider.changed.connect(self.update_vertical_contour_min_length)

        self.find_vertical_lines_button = QPushButton("Find Vertical Lines")
        self.find_vertical_lines_button.clicked.connect(self.activate_find_vertical_lines)
        """

        self.find_tables_button = QPushButton("Find Tables")
        self.find_tables_button.clicked.connect(self.activate_find_tables)

        VBlayout = QVBoxLayout(self)
        # VBlayout.addWidget(self.viewer)
        # HBlayout = QHBoxLayout()
        VBlayout.addWidget(self.loadImageButton)

        VBlayout.addWidget(self.horizontal_contour_min_length_slider)
        VBlayout.addWidget(self.find_horizontal_lines_button)

        # VBlayout.addWidget(self.vertical_contour_min_length_slider)
        # VBlayout.addWidget(self.find_vertical_lines_button)
        VBlayout.addWidget(self.find_tables_button)
        self.table = None

        self.image_viewer = None
        self.horizontal_viewer = None
        # self.vertical_viewer = None
        self.table_viewer = None

        self.equalized_image = None
        # HBlayout.setAlignment(Qt.AlignLeft)
        # HBlayout.addWidget(self.loadImageButton)
        # VBlayout.addLayout(HBlayout)

    """
    def photoClicked(self, pos):
        if self.viewer.dragMode() == QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))
    """

    def update_horizontal_contour_min_length(self, value):
        self.horizontal_contour_min_length = value

    """
    def update_vertical_contour_min_length(self, value):
        self.vertical_contour_min_length = value
    
    #image, equalized_image, horizontal_contours, horizontal_contour_min_size, horizontal_lines, 
                # horizontal_lines_image
    """

    def initiate_worker(self):
        self.worker = Worker(self.image, self.equalized_image, self.gray, self.horizontal_contours,
                             self.horizontal_contour_min_length,
                             self.horizontal_lines, self.horizontal_lines_image)

        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.finished.connect(self.worker_thread.quit)

    def activate_get_image(self):
        if not self.operationLock.tryLock():
            return
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File',
                                                   r"C:\school\Herr Eugen Antal_2020-09-23_220056\tables",
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
        self.worker.processed3.connect(self.image_processed)
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

    def image_processed(self, equalized_image, gray, horizontal_contours):
        self.horizontal_contours = horizontal_contours
        self.gray = gray
        self.equalized_image = equalized_image
        self.save_image(equalized_image, "equalized")
        QMessageBox.about(self, "Title", "Finished Processing")
        self.image_viewer = create_viewer(self.image, "Image")
        # self.viewer.setPhoto(QPixmap( QPixmap.fromImage(toQImage(thinned))))
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
        self.worker.processed_lines_image.connect(self.finished_finding_horizontal_lines)
        self.worker_thread.start()

    def finished_finding_horizontal_lines(self, horizontal_lines, horizontal_lines_image, image):
        self.horizontal_image = image
        self.horizontal_lines = horizontal_lines
        self.horizontal_lines_image = horizontal_lines_image
        self.save_image(image, "horizontal-lines-found")
        self.operationLock.unlock()

        QMessageBox.about(self, "Title", "Found Horizontal Lines")
        self.horizontal_viewer = create_viewer(image, "Horizontal Lines Detected")
        # print(str(self.horizontal_contour_min_length))
        # print(str(self.horizontal_contour_gap))


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

        self.initiate_worker()
        self.worker_thread.started.connect(self.worker.find_tables)
        self.worker.processed_list_image.connect(self.finished_finding_tables)
        self.worker_thread.start()

    def finished_finding_tables(self, table, image):
        self.table_viewer = create_viewer(image, "Higlated Table")
        self.save_image(image, "table-" + str(self.horizontal_contour_min_length) + "-")
        self.save_variable(table, "table-" + str(self.horizontal_contour_min_length) + "-")
        self.table = table
        self.operationLock.unlock()


class Worker(QObject):
    processed1 = pyqtSignal(np.ndarray)
    processed2 = pyqtSignal(np.ndarray, np.ndarray)
    processed3 = pyqtSignal(np.ndarray, np.ndarray, list)
    processed4 = pyqtSignal(np.ndarray, list)
    processed5 = pyqtSignal(dict)
    processed6 = pyqtSignal(list)

    processed_lines_image = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    processed_2_images = pyqtSignal(np.ndarray, np.ndarray)
    processed_list_image = pyqtSignal(list, np.ndarray)
    finished = pyqtSignal()

    def __init__(self, image, equalized_image, gray, horizontal_contours, horizontal_contour_min_size, horizontal_lines,
                 horizontal_lines_image):
        super().__init__()
        self.image = image
        self.equalized_image = equalized_image
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.horizontal_contours = horizontal_contours
        self.horizontal_contour_min_size = horizontal_contour_min_size
        self.horizontal_lines = horizontal_lines
        self.horizontal_lines_image = horizontal_lines_image
        self.gray = gray

    def preprocess_image(self):
        if self.image is not None:
            print(self.image.shape)
            equalized_image, gray = preprocess_img(self.image)
            horizontal_contours = get_horizontal_contours(equalized_image)
            # horizontal_contours, vertical_contours = preprocess_img(self.image)
            self.processed3.emit(equalized_image, gray, horizontal_contours)
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
    """
    def find_vertical_lines(self):

        img = np.zeros((self.image_height, self.image_width), np.uint8)
        filtered_vertical_contours = []
        for contour in self.vertical_contours:
            if contour.shape[0] > self.vertical_contours_size:
                filtered_vertical_contours.append(contour)

        cv2.drawContours(img, filtered_vertical_contours, -1, (255, 255, 255))

        img2 = np.copy(self.image)
        cv2.drawContours(img2, filtered_vertical_contours, -1, (255, 0, 0), 4)
        # first is just drawn vertical controures, second is contoures drawn contoures onto original image
        self.processed_2_images.emit(img, img2)
        self.finished.emit()
    """

    def find_tables(self):
        table = []
        available_area_minimum = 80
        minimum_table_width = 100
        overlap_margin = 50
        for i in range(0, self.horizontal_lines.shape[0] - 1):
            table_line = []
            j = i + 1
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
                    if overlap_start_point < overlap_end_point and (
                            overlap_end_point - overlap_start_point) > minimum_table_width:
                        if (overlap_end_point - overlap_start_point) > available_area:
                            available_area = 0
                        else:
                            available_area = available_area - (overlap_end_point - overlap_start_point)
                        overlapping_points.append(np.array([j, overlap_start_point, overlap_end_point], np.uint32))
                        found_next_line = True
                j += 1
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
                vert_areas, vertical_lines_image = find_vertical_lines_between_horizontal_lines(self.equalized_image,
                                                                                                self.horizontal_lines_image,
                                                                                                self.horizontal_lines[
                                                                                                    i],
                                                                                                self.horizontal_lines[
                                                                                                    overlap[0]],
                                                                                                tmp_overlap_1,
                                                                                               tmp_overlap_2)
                if vert_areas is not None:
                    for j in range(0, vert_areas.shape[0] - 1):
                        pt_1_y = self.horizontal_lines[i][0]
                        pt_2_y = self.horizontal_lines[i][0]

                        pt_3_y = self.horizontal_lines[overlap[0]][1]
                        pt_4_y = self.horizontal_lines[overlap[0]][1]

                        # rotated_horizontal_image = rotate(self.horizontal_image,self.angle_for_vertical_lines)
                        sliced_image = self.horizontal_lines_image[self.horizontal_lines[i][0]: end_row + 1,
                                       int(overlap[1] + vert_areas[j][0]): int(overlap[1] + vert_areas[j][1] + 1)]

                        sliced_horizontal_proj = horizontal_projection(sliced_image)

                        sliced_areas = find_areas(sliced_horizontal_proj, 1, 1)
                        if sliced_areas.shape[0] > 0:
                            pt_1_y = sliced_areas[0][0] + self.horizontal_lines[i][0]
                            pt_3_y = sliced_areas[-1][1] + self.horizontal_lines[i][0]

                        sliced_image = self.horizontal_lines_image[self.horizontal_lines[i][0]: end_row,
                                       vert_areas[j + 1][0]: vert_areas[j + 1][1] + 1]

                        sliced_horizontal_proj = horizontal_projection(sliced_image)

                        sliced_areas = find_areas(sliced_horizontal_proj, 1, 1)

                        if sliced_areas.shape[0] > 0:
                            pt_2_y = sliced_areas[0][0] + self.horizontal_lines[i][0]
                            pt_4_y = sliced_areas[-1][1] + self.horizontal_lines[i][0]

                        if pt_1_y > self.horizontal_lines[i][1] or pt_1_y < self.horizontal_lines[i][0]:
                            pt_1_y = self.horizontal_lines[i][0]

                        if pt_2_y > self.horizontal_lines[i][1] or pt_2_y < self.horizontal_lines[i][0]:
                            pt_2_y = self.horizontal_lines[i][0]

                        if pt_3_y < self.horizontal_lines[overlap[0]][0] or pt_3_y > end_row:
                            pt_3_y = self.horizontal_lines[overlap[0]][1]

                        if pt_4_y < self.horizontal_lines[overlap[0]][0] or pt_4_y > end_row:
                            pt_4_y = self.horizontal_lines[overlap[0]][1]

                        # try to recover x coordinates

                        pt_1_x = vert_areas[j][0] + overlap[1]
                        pt_3_x = vert_areas[j][0] + overlap[1]

                        pt_2_x = vert_areas[j + 1][1] + overlap[1]
                        pt_4_x = vert_areas[j + 1][1] + overlap[1]

                        sliced_image = vertical_lines_image[self.horizontal_lines[i][0]: self.horizontal_lines[i][1],
                                       vert_areas[j][0]: vert_areas[j + 1][1] + 1]

                        sliced_vertical_projection = vertical_projection(sliced_image)
                        sliced_areas = find_areas(sliced_vertical_projection, 1, 1)
                        if sliced_areas.shape[0] > 0:
                            pt_1_x = sliced_areas[0][0] + vert_areas[j][0] + overlap[1]
                            pt_2_x = sliced_areas[-1][1] + vert_areas[j][0] + overlap[1]

                        sliced_image = vertical_lines_image[self.horizontal_lines[overlap[0]][0]: end_row,
                                       vert_areas[j][0]: vert_areas[j + 1][1] + 1]

                        sliced_vertical_projection = vertical_projection(sliced_image)
                        sliced_areas = find_areas(sliced_vertical_projection, 1, 1)

                        if sliced_areas.shape[0] > 0:
                            pt_3_x = sliced_areas[0][0] + vert_areas[j][0] + overlap[1]
                            pt_4_x = sliced_areas[-1][1] + vert_areas[j][0] + overlap[1]

                        if pt_1_x > (overlap[1] + vert_areas[j][1]) or pt_1_x < (overlap[1] + vert_areas[j][0]):
                            pt_1_x = overlap[1] + vert_areas[j][0]

                        if pt_2_x < (overlap[1] + vert_areas[j + 1][0]) or pt_2_x > (overlap[1] + vert_areas[j + 1][1]):
                            pt_2_x = overlap[1] + vert_areas[j + 1][1]

                        if pt_3_x > (overlap[1] + vert_areas[j][1]) or pt_3_x < (overlap[1] + vert_areas[j][0]):
                            pt_3_x = overlap[1] + vert_areas[j][0]

                        if pt_4_x < (overlap[1] + vert_areas[j + 1][0]) or pt_4_x > (overlap[1] + vert_areas[j + 1][1]):
                            pt_4_x = overlap[1] + vert_areas[j + 1][1]

                        table_line.append(
                            [(pt_1_x, pt_1_y), (pt_2_x, pt_2_y), (pt_3_x, pt_3_y), (pt_4_x, pt_4_y), pt_2_x - pt_1_x,
                             pt_4_y - pt_2_y])

            table.append(table_line)

        table = try_fitting_table(table)

        classification_string, classification_image = classify_table(table, self.horizontal_lines, self.gray, np.copy(self.image))
        print(classification_string)
        cv2.imwrite("export/classified-image.jpg",classification_image)
        img = np.copy(self.image)
        used_colors = []
        if len(table) > 0:

            # img = rotate(img, self.min_horizontal_area['angle'])
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

        self.processed_list_image.emit(table, img)
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
        img = np.zeros((self.image_height, self.image_width), np.uint8)
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
            vert_areas = find_areas(vert_proj, 2, 1)
            horizontal_line_areas.append(np.append(area, [vert_areas[0][0], vert_areas[-1][1]]))
        horizontal_line_areas = np.array(horizontal_line_areas)
        img2 = np.copy(self.image)
        if horizontal_line_areas.shape[0]> 0:
            start_point = horizontal_line_areas[np.argsort(horizontal_line_areas[:, 3])][0, 3]
            end_point = horizontal_line_areas[np.argsort(horizontal_line_areas[:, 4])][-1, 4]

            for area in horizontal_line_areas:
                if area[3] < (start_point + 100):
                    area[3] = start_point
                if area[4] < (end_point - 100):
                    area[4] = end_point


            for area in horizontal_line_areas:
                cv2.line(img2, (area[3], area[0]), (area[4], area[0]), (255, 0, 0), 4)
                cv2.line(img2, (area[3], area[1]), (area[4], area[1]), (0, 255, 0), 4)

        self.processed_lines_image.emit(horizontal_line_areas, img, img2)
        # self.processed5.emit(min_area)
        self.finished.emit()


def find_vertical_lines_between_horizontal_lines(equalized_image, horizontal_lines_image, horizontal_line_area_1,
                                                 horizontal_line_area_2, start_x, end_x):
    # print("test")

    y_margin = 10
    # try to look up and down from the lines - vertical lines usually go up and down
    start_row = horizontal_line_area_1[0] - y_margin
    start_row = 0 if start_row < 0 else start_row
    end_row = horizontal_line_area_2[1] + y_margin + 1
    end_row = equalized_image.shape[0] if end_row > equalized_image.shape[0] else end_row
    tmp_image = equalized_image[start_row: end_row, start_x: end_x + 1]
    cont_v = get_vertical_contours(tmp_image)

    filtered_contours = []

    for cont in cont_v:
        if cont.shape[0] >= int(horizontal_line_area_2[0] - horizontal_line_area_1[1]) - \
                int((horizontal_line_area_2[0] - horizontal_line_area_1[1]) / 20):
            filtered_contours.append(cont)

    classified_contours_v = contour_classification(filtered_contours)
    classified_lines_v = find_line_fragments_2(classified_contours_v, 1, 0)

    contoured_vertical_image = np.zeros_like(tmp_image)
    for c_line in classified_lines_v:
        if c_line.classification == 3:
            for fragment in c_line.fragments:
                cv2.line(contoured_vertical_image, fragment.start_point, fragment.end_point, (255, 255, 255), 1)

    # debug
    # cv2.imwrite("export/cont.jpg",contoured_vertical_image)

    vert_proj = vertical_projection(contoured_vertical_image)
    vert_areas = find_areas(vert_proj, 10, 1)
    if vert_areas.shape[0] == 0:
        return None, contoured_vertical_image

    # debug
    """
    contoured_image_2_debug = cv2.cvtColor(contoured_vertical_image, cv2.COLOR_GRAY2BGR)
    for area in vert_areas:
        cv2.line(contoured_image_2_debug, (area[0],0), (area[0],contoured_image_2_debug.shape[0]-1), (255,0,0),2)
        cv2.line(contoured_image_2_debug, (area[1],0), (area[1], contoured_image_2_debug.shape[0]-1), (0,255,0),2)
    cv2.imwrite("export/cont2.jpg", contoured_image_2_debug)
    """
    # debug

    vert_areas_tmp = []
    for area in vert_areas:
        tmp_horizontal_proj = horizontal_projection(contoured_vertical_image[:, area[0]: area[1] + 1])

        # debug
        # cv2.imwrite("export/vert-line-horizontal.jpg",contoured_vertical_image[:, area[0]: area[1] + 1])

        tmp_horizontal_areas = find_areas(tmp_horizontal_proj, 1,
                                          12)

        if tmp_horizontal_areas.shape[0] == 0:
            continue
        else:
            if tmp_horizontal_areas.shape[0] == 1:
                horizontal_area_length = tmp_horizontal_areas[0, 2]
            else:
                # should this be here?
                horizontal_area_length = int(tmp_horizontal_areas[-1][1] - tmp_horizontal_areas[0][0])
            # if horizontal_area_length >= int(self.horizontal_lines[overlap[0]][0] - self.horizontal_lines[i][1]):
            vert_areas_tmp.append(np.append(area, [horizontal_area_length]))

    vert_areas = np.array(vert_areas_tmp)
    vert_areas_tmp = []

    for vert_area in vert_areas:

        vert_area_x_2 = vert_area[1] + 30
        if vert_area_x_2 >= horizontal_lines_image.shape[1]:
            vert_area_x_2 = horizontal_lines_image.shape[1] - 1

        tmp_horizontal_image = horizontal_lines_image[horizontal_line_area_1[0]: horizontal_line_area_2[1] + 1,
                               vert_area[0]: vert_area_x_2]
        # debug
        # cv2.imwrite("export/aaa.jpg",tmp_horizontal_image)

        # debug
        """
        horizontal_line_image_2 = np.copy(horizontal_lines_image)
        horizontal_line_image_2 = cv2.cvtColor(horizontal_line_image_2, cv2.COLOR_GRAY2BGR)
        cv2.line(horizontal_line_image_2,(vert_area[0],horizontal_line_area_1[0]), (vert_area[0], horizontal_line_area_2[1]), (255,0,0),2)
        cv2.line(horizontal_line_image_2, (vert_area[1], horizontal_line_area_1[0]), (vert_area[1], horizontal_line_area_2[1]), (0,255,0),2)
        cv2.imwrite("export/bbb.jpg", horizontal_line_image_2)
        """
        # debug

        tmp_horizontal_proj = horizontal_projection(tmp_horizontal_image)
        tmp_horizontal_areas = find_areas(tmp_horizontal_proj, 2, 1)
        if tmp_horizontal_areas.shape[0] == 0:
            continue
            # min_length = horizontal_line_area_2[0] - horizontal_line_area_1[1]
        elif tmp_horizontal_areas.shape[0] == 1:
            if tmp_horizontal_areas[0][0] > horizontal_line_area_2[0] - horizontal_line_area_1[0]:
                min_length = horizontal_line_area_1[0] + tmp_horizontal_areas[0][1] - horizontal_line_area_1[1]
            else:
                min_length = horizontal_line_area_2[0] - horizontal_line_area_1[0] - tmp_horizontal_areas[0][0]
        else:
            min_length = tmp_horizontal_areas[-1][1] - tmp_horizontal_areas[0][0]
        # add some percentage?
        min_length -= int(min_length / 10)
        if vert_area[3] >= min_length:
            vert_areas_tmp.append(vert_area)

    vert_areas = np.array(vert_areas_tmp)

    return vert_areas, contoured_vertical_image


def try_fitting_table(table):
    for i in range(0, len(table)):
        if len(table[i]) > 0:
            changed = True
            while changed:
                changed = False
                values_list = []
                for t in table[i]:
                    values_list.append(t[4])
                values_list = np.array(values_list)
                width_mean = np.mean(values_list)
                width_std = np.std(values_list)
                cut_off = width_std * 1.5
                lower = width_mean - cut_off
                upper = width_mean + cut_off
                for j in range(0, len(table[i])):
                    if table[i][j][4] < lower:
                        # merge two next ??
                        # needs to check if merging left or right is below upper?
                        if j > 0:
                            if table[i][j][1][0] - table[i][j - 1][0][0] < upper:
                                # merging those two tables together
                                table_line_tmp = table[i][0:j - 1]
                                table_line_tmp.append(
                                    [
                                        table[i][j - 1][0],
                                        table[i][j][1],
                                        table[i][j - 1][2],
                                        table[i][j][3],
                                        table[i][j][1][0] - table[i][j - 1][0][0],
                                        table[i][j][5]
                                    ]
                                )
                                if j + 1 < len(table[i]):
                                    for k in range(j + 1, len(table[i])):
                                        table_line_tmp.append(table[i][k])
                                table[i] = table_line_tmp
                                changed = True
                                break
                        if j + 1 < len(table[i]):
                            if table[i][j + 1][1][0] - table[i][j][0][0] < upper:
                                table_line_tmp = table[i][0:j]
                                table_line_tmp.append(
                                    [
                                        table[i][j][0],
                                        table[i][j + 1][1],
                                        table[i][j][2],
                                        table[i][j + 1][3],
                                        table[i][j + 1][1][0] - table[i][j][0][0],
                                        table[i][j][5]
                                    ]
                                )
                                if j + 2 < len(table[i]):
                                    for k in range(j + 2, len(table[i])):
                                        table_line_tmp.append(table[i][k])
                                table[i] = table_line_tmp
                                changed = True
                                break
                        """       
                        if table[i][j+1][4] < lower:
                            # merge them:
                            table_line_tmp = table[i][0:j]
                            table_line_tmp.append(
                                [
                                    table[i][j][0],
                                    table[i][j+1][1],
                                    table[i][j][2],
                                    table[i][j+1][3],
                                    table[i][j+1][1][0] - table[i][j][0][0],
                                    table[i][j][5]])

                            #try appending rest of the list:
                            if j+2 < len(table[i]):
                                for k in range(j+2, len(table[i])):
                                    table_line_tmp.append(table[i][k])
                            table[i] = table_line_tmp
                            changed = True
                            break
                        """
                    elif table[i][j][4] > upper:
                        if table[i][j][1][0] - (table[i][j][0][0] + width_mean) > lower:
                            table_line_tmp = table[i][0:j]
                            table_line_tmp.append(
                                [
                                    table[i][j][0],
                                    (int(table[i][j][0][0] + width_mean), table[i][j][0][1]),
                                    table[i][j][2],
                                    (int(table[i][j][3][0] + width_mean), table[i][j][3][1]),
                                    int(width_mean),
                                    table[i][j][5]
                                ]
                            )
                            table_line_tmp.append(
                                [
                                    (int(table[i][j][0][0] + width_mean), table[i][j][0][1]),
                                    table[i][j][1],
                                    (int(table[i][j][2][0] + width_mean), table[i][j][2][1]),
                                    table[i][j][3],
                                    int(width_mean),
                                    table[i][j][5]
                                ]
                            )

                            if j + 2 < len(table[i]):
                                for k in range(j + 2, len(table[i])):
                                    table_line_tmp.append(table[i][k])
                            changed = True
                            table[i] = table_line_tmp
                            break
                """
                if changed is False and table[i][-1][4] > upper:
                    if table[i][-1][1][0] - (table[i][-1][0][0] + width_mean) > lower:
                        table_line_tmp = table[i][0:-1]
                        table_line_tmp.append(
                            [
                                table[i][-1][0],
                                (int(table[i][-1][0][0] + width_mean), table[i][-1][0][1]),
                                table[i][-1][2],
                                (int(table[i][-1][3][0] + width_mean), table[i][-1][3][1]),
                                int(width_mean),
                                table[i][-1][5]
                            ]
                        )
                        table_line_tmp.append(
                            [
                                (int(table[i][-1][0][0] + width_mean), table[i][-1][0][1]),
                                table[i][-1][1],
                                (int(table[i][-1][2][0] + width_mean), table[i][-1][2][1]),
                                table[i][-1][3],
                                int(width_mean),
                                table[i][-1][5]
                            ]
                        )
                        table[i] = table_line_tmp
                        changed = True
    """
    # try to see if we need to add left or right column?
    return table


def classify_table(table, horizontal_lines, gray, image):
    if table is None:
        return None
    table_means = []
    for i in range(0, len(table)):
        if len(table[i]) > 0:
            x_values = []
            y_values = []
            for j in range(0, len(table[i])):
                x_values.append(table[i][j][4])
                y_values.append(table[i][j][5])

            x_values = np.array(x_values)
            y_values = np.array(y_values)

            x_means = np.mean(x_values)
            y_means = np.mean(y_values)

            table_means.append([x_means, y_means])

        else:
            table_means.append([0, 0])

    table_classification = []
    header_body_combo = True

    delta = 0.2

    i = 0
    skipped  = False
    header_bodies = []
    contents = []

    if len(table)> 1:
        if len(table[0]) < 4:
            i = 1
            #skip wrong header
    while header_body_combo and i + 1 < len(table):
        if len(table[i]) == 0:
            contents.append(i)
            i+=1
            continue
        upper_1 = table_means[i][0] + (table_means[i][0] * delta)
        lower_1 = table_means[i][0] - (table_means[i][0] * delta)

        if lower_1 < table_means[i + 1][0] < upper_1:
            header_bodies.append(i)
            i += 2
        else:
            contents.append(i)
            i += 1

            header_body_combo = False
    while i < len(table):
        contents.append(i)
        i += 1

    cipher_types = {"subs": False, "bigrams": False, "nulls": False, "codeBook": False, "inverse":False}
    # now try to see if header bodies are the same vs content are the same

    same_header_body = []
    same_header_body_line = [0]
    if len(header_bodies) > 0:
        for i in range(0, len(header_bodies) - 1):
            lower = table_means[header_bodies[i]][0] - (table_means[header_bodies[i]][0] * delta)
            upper = table_means[header_bodies[i]][0] + (table_means[header_bodies[i]][0] * delta)
            if lower < table_means[header_bodies[i + 1]][0] < upper:
                same_header_body_line.append(i + 1)
                #add_last = False
            else:
                same_header_body.append(same_header_body_line)
                same_header_body_line = [i+1]
                #add_last = True

        same_header_body.append(same_header_body_line)

        lines_classified = []
        if len(same_header_body) != 0:
            for header_body_line in same_header_body:
                column_count = 0
                for header_body in header_body_line:
                    column_count += len(table[header_bodies[header_body]])
                is_multiple = True if table_means[header_body_line[0]][1] * 2 < table_means[header_body_line[0] + 1][1] \
                    else False
                class_string = ""
                if column_count < 20:
                    if cipher_types['bigrams'] is False:
                        cipher_types['bigrams'] = True
                        class_string = "2"
                    else:
                        class_string = "0"

                elif 20 < column_count < 28:
                    if cipher_types['subs'] is False:
                        cipher_types['subs'] = True
                        class_string = "1"
                    elif cipher_types['bigrams'] is False:
                        cipher_types['bigrams'] = True
                        class_string = "2"
                    else:
                        class_string = "0"
                    if is_multiple:
                        class_string += "f"
                else:
                    if cipher_types['subs'] is False:
                        if cipher_types['bigrams'] is False:
                            cipher_types['subs'] = True
                            cipher_types['bigrams'] = True
                            class_string = "12"
                            if is_multiple:
                                class_string += "f"

                table_lines = []
                for header_body in header_body_line:
                    table_lines.append(["h", header_bodies[header_body]])
                    table_lines.append(["b", header_bodies[header_body] + 1])
                    lines_classified.append(header_bodies[header_body])
                    lines_classified.append(header_bodies[header_body] + 1)
                table_classification.append([class_string, table_lines])

    content_classification = []
    if len(contents) > 0:
        if contents[0] == 0 and len(table_classification) > 0:
            contents.pop(0)

    binarized = get_binary_after_gabor(gray)
    cv2.imwrite("export/binary.jpg", binarized)




    if len(contents) > 0:
        if len(contents) > 1:
            for i in range(0, len(contents)):
                class_string = ""

                sliced = binarized[horizontal_lines[contents[i]][1]: horizontal_lines[contents[i]+1][0],:]
                cv2.imwrite("export/sliced2.jpg",sliced)
                try:
                    line_number = get_number_of_lines(sliced)
                except IndexError:
                    line_number = 0
                if cipher_types['subs'] is False:
                    if i > 0:
                        cipher_types['subs'] = True
                    class_string = "1"
                else:
                    if line_number > 5:
                        class_string = "c/{-1}"
                    else:
                        class_string = "0"
                content_classification.append([
                    class_string,
                    contents[i]
                ])
        else:
            sliced = binarized[horizontal_lines[contents[0]][1]: horizontal_lines[contents[0] + 1][0], :]
            cv2.imwrite("export/sliced2.jpg", sliced)
            try:
                line_number = get_number_of_lines(sliced)
            except IndexError:
                line_number = 0
            if cipher_types['subs'] is False:
                if line_number > 3:
                    class_string = "c"
                else:
                    class_string = "1"
            else:
                if line_number > 5:
                    class_string = "c"
                else:
                    class_string = "0"
            content_classification.append(
                [
                    class_string,
                    contents[0]
                ]
            )


    extra_area_classification = None


    if horizontal_lines is not None and horizontal_lines.shape[0] > 0:
        start_y = horizontal_lines[-1][1]
    else:
        start_y = 0


    sliced = binarized[start_y+10:,:]
    cv2.imwrite("export/sliced.jpg",sliced)
    horizont_proj = horizontal_projection(sliced)
    horizont_areas = find_areas(horizont_proj,1,2)

    vert_proj = vertical_projection(sliced)
    vert_areas = find_areas(vert_proj,2,80)

    if vert_areas.shape[0] == 1:
        start_x = vert_areas[0][0]
        end_x = vert_areas[0][1]
    elif vert_areas.shape[0] > 1:
        start_x = vert_areas[0][0]
        end_x = vert_areas[-1][1]

    if horizont_areas.shape[0] == 1:
        start_y2 = horizont_areas[0][0]
        end_y2 = horizont_areas[0][1]
    elif horizont_areas.shape[0] > 1:
        start_y2 = horizont_areas[0][0]
        end_y2 = horizont_areas[-1][1]



    if horizont_areas.shape[0] > 0 and vert_areas.shape[0] > 0:
        try:
            n_lines = get_number_of_lines(sliced)
        except:
            n_lines = 0

        if n_lines > 6:
            class_string = "c/{-1}"
        else:
            class_string = "0"
        extra_area_classification = [
            class_string,
            (start_x, start_y+start_y2),
            (end_x, start_y+ + start_y2),
            (start_x, start_y + end_y2),
            (end_x, start_y+end_y2)
        ]


    #create string
    used_colors = []
    classification_string = ""

    if len(table_classification)  > 0:
        for table_class in table_classification:
            classification_string += table_class[0]
            classification_string+= "\n"

            start_x = table[table_class[1][0][1]][0][0][0]
            end_x = table[table_class[1][0][1]][-1][1][0]

            start_y = table[table_class[1][0][1]][0][0][1]
            end_y = table[table_class[1][-1][1]][0][2][1]

            for table_class_line in table_class[1]:

                if table[table_class_line[1]][0][0][0] < start_x:
                    start_x = table[table_class_line[1]][0][0][0]
                if table[table_class_line[1]][0][2][0] < start_x:
                    start_x = table[table_class_line[1]][0][2][0]

                if table[table_class_line[1]][-1][1][0] > end_x:
                    end_x = table[table_class_line[1]][-1][1][0]

                if table[table_class_line[1]][-1][3][0] > end_x:
                    end_x = table[table_class_line[1]][-1][3][0]

                classification_string+= table_class_line[0]
                classification_string += " "
                classification_string+= str(table[table_class_line[1]][0][0])
                classification_string+=" "
                classification_string+= str(table[table_class_line[1]][-1][1])
                classification_string+=" "
                classification_string+= str(table[table_class_line[1]][0][2])
                classification_string+= " "
                classification_string+= str(table[table_class_line[1]][-1][3])
                classification_string+="\n"
            classification_string+="\n"

            color = list(np.random.random(size=3) * 256)
            while color in used_colors:
                color = list(np.random.random(size=3) * 256)
            used_colors.append(color)
            cv2.line(image, (start_x,start_y), (end_x,start_y), color, 4)
            cv2.line(image, (start_x, start_y), (start_x,end_y), color, 4)
            cv2.line(image, (end_x, start_y), (end_x,end_y), color, 4)
            cv2.line(image, (start_x,end_y), (end_x, end_y), color, 4)

    if len(content_classification) > 0:
        for cont in content_classification:
            classification_string+= cont[0]
            classification_string+="\n"
            start_y = horizontal_lines[cont[1]][0]
            end_y = horizontal_lines[cont[1]+1][1]
            if len(table[cont[1]]) == 0:
                start_x = horizontal_lines[cont[1]][3]
                end_x = horizontal_lines[cont[1]][4]

            else:
                start_x = table[cont[1]][0][0][0]
                end_x = table[cont[1]][-1][1][0]

            color = list(np.random.random(size=3) * 256)
            while color in used_colors:
                color = list(np.random.random(size=3) * 256)
            used_colors.append(color)

            cv2.line(image, (start_x,start_y), (end_x,start_y), color, 4)
            cv2.line(image, (start_x,start_y), (start_x, end_y), color, 4)
            cv2.line(image, (end_x,start_y), (end_x,end_y), color, 4)
            cv2.line(image, (start_x, end_y), (end_x,end_y), color, 4)

            classification_string+="c "
            classification_string+= str((start_x,start_y))
            classification_string+= " "
            classification_string+= str((end_x,start_y))
            classification_string+= " "
            classification_string+= str((start_x,end_y))
            classification_string+= " "
            classification_string+=str((end_x,end_y))
            classification_string+="\n"
        classification_string+="\n"

    if extra_area_classification is not None:

        color = list(np.random.random(size=3) * 256)
        while color in used_colors:
            color = list(np.random.random(size=3) * 256)
        used_colors.append(color)

        cv2.line(image, extra_area_classification[1], extra_area_classification[2], color, 4)
        cv2.line(image, extra_area_classification[1], extra_area_classification[3], color, 4)
        cv2.line(image, extra_area_classification[2], extra_area_classification[4], color, 4)
        cv2.line(image, extra_area_classification[3], extra_area_classification[4], color, 4)

        classification_string+= extra_area_classification[0]
        classification_string+= "\n"
        classification_string+= "c "
        classification_string+= str(extra_area_classification[1])
        classification_string+= " "
        classification_string+= str(extra_area_classification[2])
        classification_string+= " "
        classification_string+=str(extra_area_classification[3])
        classification_string+= " "
        classification_string+= str(extra_area_classification[4])
        classification_string+="\n"

    return classification_string, image



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
