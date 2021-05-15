import cv2
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import time

from image_preprocessor import preprocess_img, get_horizontal_contours_gabor
from projections import horizontal_projection, find_areas, vertical_projection
from nomenclator_classificator import find_vertical_lines_between_horizontal_lines, classify_table, try_fitting_table


class Worker(QObject):

    finished = pyqtSignal()
    send_image_image_list = pyqtSignal(np.ndarray, np.ndarray, list)
    detected_horizontal_lines = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    detected_tables = pyqtSignal(list, np.ndarray, str, np.ndarray)

    def __init__(self, image, export_folder, is_debug, horizontal_contours, horizontal_contour_min_size,
                 horizontal_line_areas, horizontal_contoured_image, gray, equalized):
        super().__init__()
        self.image = image
        self.image_height, self.image_width = image.shape[:2]
        self.export_folder = export_folder
        self.is_debug = is_debug
        self.horizontal_contours = horizontal_contours
        self.horizontal_contour_min_size = horizontal_contour_min_size
        self.horizontal_line_areas = horizontal_line_areas
        self.horizontal_contoured_image = horizontal_contoured_image
        self.gray = gray
        self.equalized = equalized

    def preprocess_image(self):
        equalized, gray = preprocess_img(self.image)
        horizontal_contours = get_horizontal_contours_gabor(equalized)
        self.send_image_image_list.emit(equalized, gray, horizontal_contours)
        self.finished.emit()

    def find_horizontal_lines(self):
        img = np.zeros((self.image_height, self.image_width), np.uint8)

        # filter contours
        filtered_horizontal_contours = []
        for contour in self.horizontal_contours:
            # length of contour has to be higher than minimal length
            if contour.shape[0] > self.horizontal_contour_min_size:
                filtered_horizontal_contours.append(contour)

        cv2.drawContours(img, filtered_horizontal_contours, -1, (255, 255, 255))
        horizontal_proj = horizontal_projection(img)
        # finding horizontal slice of image where line is found
        horizontal_areas = find_areas(horizontal_proj, 10, 1)
        # bounding boxes for horizontal lines
        horizontal_line_areas = []
        for area in horizontal_areas:
            # find start_x and end_x coordinates based on vertical projection
            # slice out where line lies to find its edges
            sliced_vertical_img = img[area[0]:area[1], :]
            vert_proj = vertical_projection(sliced_vertical_img)
            vert_areas = find_areas(vert_proj, 2, 1)
            # start_x is  beginning of leftmost continuous area, end_x is  end of rightmost continuous area
            horizontal_line_areas.append(np.append(area, [vert_areas[0][0], vert_areas[-1][1]]))

        # transform into numpy array for ease of work
        horizontal_line_areas = np.array(horizontal_line_areas)

        # draw out detected lines to uses can see and evaluate if the detection is correct?
        img2 = np.copy(self.image)
        if horizontal_line_areas.shape[0] > 0:
            start_point = horizontal_line_areas[np.argsort(horizontal_line_areas[:, 3])][0, 3]
            end_point = horizontal_line_areas[np.argsort(horizontal_line_areas[:, 4])][-1, 4]

            # try correcting lines - if they are too short maybe there is degradation?
            for area in horizontal_line_areas:

                area[3] = start_point

                area[4] = end_point

            for area in horizontal_line_areas:
                cv2.line(img2, (area[3], area[0]), (area[4], area[0]), (255, 0, 0), 4)
                cv2.line(img2, (area[3], area[1]), (area[4], area[1]), (0, 255, 0), 4)

        self.detected_horizontal_lines.emit(horizontal_line_areas,img,img2)
        self.finished.emit()

    def find_table(self):
        start_time = time.time()
        table = []
        available_area_minimum = 80
        minimum_table_width = 100
        overlap_margin = 50
        x_margin = 50
        start_x = self.horizontal_line_areas[0][3] - x_margin
        if start_x < 0:
            start_x = 0
        # we go from top to bottom and search between 2 lines
        for i in range(0, self.horizontal_line_areas.shape[0] - 1):
            # line in the table
            table_line = []
            x_margin = 50
            #detecting vertical lines
            vert_areas, vertical_lines_image = \
                find_vertical_lines_between_horizontal_lines(self.equalized, self.horizontal_contoured_image,
                                                             self.horizontal_line_areas[i],
                                                             self.horizontal_line_areas[i+1], start_x)
            if vert_areas is not None:
                #end in y
                end_row = self.horizontal_line_areas[i+1][1]

                #start in x
                #start_x = self.horizontal_line_areas[i][3]

                # go over all vertical lines detected
                for j in range(0, vert_areas.shape[0] - 1):

                    # set y of top points to be top most y where horizontal line could be
                    pt_1_y = self.horizontal_line_areas[i][0]
                    pt_2_y = self.horizontal_line_areas[i][0]

                    # set bottom y to most bottom part where horizontal line could be
                    pt_3_y = self.horizontal_line_areas[i+1][1]
                    pt_4_y = self.horizontal_line_areas[i+1][1]

                    #  slice horizontal image and check where the actual coordinates lie
                    sliced_image = self.horizontal_contoured_image[self.horizontal_line_areas[i][0]: end_row + 1,
                                   int(start_x + vert_areas[j][0]): int(start_x + vert_areas[j][1] + 1)]

                    sliced_horizontal_proj = horizontal_projection(sliced_image)

                    sliced_areas = find_areas(sliced_horizontal_proj, 1, 1)
                    if sliced_areas.shape[0] > 0:
                        pt_1_y = sliced_areas[0][0] + self.horizontal_line_areas[i][0]
                        pt_3_y = sliced_areas[-1][1] + self.horizontal_line_areas[i][0]

                    sliced_image = self.horizontal_contoured_image[self.horizontal_line_areas[i][0]: end_row+1,
                                   int(start_x + vert_areas[j + 1][0]): int(start_x + vert_areas[j + 1][1]) + 1]

                    sliced_horizontal_proj = horizontal_projection(sliced_image)

                    sliced_areas = find_areas(sliced_horizontal_proj, 1, 1)

                    if sliced_areas.shape[0] > 0:
                        pt_2_y = sliced_areas[0][0] + self.horizontal_line_areas[i][0]
                        pt_4_y = sliced_areas[-1][1] + self.horizontal_line_areas[i][0]

                    if pt_1_y > self.horizontal_line_areas[i][1] or pt_1_y < self.horizontal_line_areas[i][0]:
                        pt_1_y = self.horizontal_line_areas[i][0]

                    if pt_2_y > self.horizontal_line_areas[i][1] or pt_2_y < self.horizontal_line_areas[i][0]:
                        pt_2_y = self.horizontal_line_areas[i][0]

                    if pt_3_y < self.horizontal_line_areas[i+1][0] or pt_3_y > end_row:
                        pt_3_y = self.horizontal_line_areas[i+1][1]

                    if pt_4_y < self.horizontal_line_areas[i+1][0] or pt_4_y > end_row:
                        pt_4_y = self.horizontal_line_areas[i+1][1]

                    pt_1_x = vert_areas[j][0] + start_x
                    pt_3_x = vert_areas[j][0] + start_x
                    pt_2_x = vert_areas[j + 1][1] + start_x
                    pt_4_x = vert_areas[j + 1][1] + start_x


                    sliced_image = vertical_lines_image[self.horizontal_line_areas[i][0]: self.horizontal_line_areas[i][1],
                                     start_x + vert_areas[j][0]: start_x+ vert_areas[j + 1][1] + 1]

                    sliced_vertical_projection = vertical_projection(sliced_image)
                    sliced_areas = find_areas(sliced_vertical_projection, 1, 1)
                    if sliced_areas.shape[0] > 1:
                        pt_1_x = sliced_areas[0][0] + vert_areas[j][0] + start_x
                        pt_2_x = sliced_areas[-1][1] + vert_areas[j][0] + start_x


                    sliced_image = vertical_lines_image[self.horizontal_line_areas[i+1][0]: end_row,
                                   start_x+vert_areas[j][0]: start_x + vert_areas[j + 1][1] + 1]

                    sliced_vertical_projection = vertical_projection(sliced_image)
                    sliced_areas = find_areas(sliced_vertical_projection, 1, 1)
                    
                    if sliced_areas.shape[0] > 0:
                        pt_3_x = sliced_areas[0][0] + vert_areas[j][0] + start_x
                        pt_4_x = sliced_areas[-1][1] + vert_areas[j][0] + start_x

                    if pt_1_x > (start_x + vert_areas[j][1]) or pt_1_x < (start_x + vert_areas[j][0]):
                        pt_1_x = start_x + vert_areas[j][0]

                    if pt_2_x < (start_x + vert_areas[j + 1][0]) or pt_2_x > (start_x + vert_areas[j + 1][1]):
                        pt_2_x = start_x + vert_areas[j + 1][1]

                    if pt_3_x > (start_x + vert_areas[j][1]) or pt_3_x < (start_x + vert_areas[j][0]):
                        pt_3_x = start_x + vert_areas[j][0]

                    if pt_4_x < (start_x + vert_areas[j + 1][0]) or pt_4_x > (start_x + vert_areas[j + 1][1]):
                        pt_4_x = start_x + vert_areas[j + 1][1]

                    table_line.append(
                        [(pt_1_x, pt_1_y), (pt_2_x, pt_2_y), (pt_3_x, pt_3_y), (pt_4_x, pt_4_y), pt_2_x - pt_1_x,
                         pt_4_y - pt_2_y])

            table.append(table_line)
        table = try_fitting_table(table)
        classification_string, classification_image = classify_table(table, self.horizontal_line_areas, self.gray,
                                                                     np.copy(self.image))

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

        print("--- %s seconds ---" % (time.time() - start_time))
        self.detected_tables.emit(table, img, classification_string, classification_image)
        self.finished.emit()