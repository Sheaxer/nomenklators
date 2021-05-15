import numpy as np
import cv2

class ClassifiedContourFragment:
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
        #go until we reach the end of the contour
        while j < contour.shape[0]:
            #x,y uses opencv coordinates
            #opencv returns [[x,y]] for contour point
            diff_x = round(contour[j][0][0] - contour[i][0][0]) + 5
            diff_y = round(contour[j][0][1] - contour[i][0][1]) + 5
            #accessing array - x are colums, y are rows, because of opencv
            classification = classification_matrix[diff_y][diff_x]
            #fragment with classified direction, contain start and end point
            classfied_fragment = ClassifiedContourFragment(classification,
                                                           (contour[i][0][0], contour[i][0][1]),
                                                           (contour[j][0][0], contour[j][0][1]))
            classified_fragments.append(classfied_fragment)
            i = j
            j += 5
        j = contour.shape[0] - 1

        diff_x = round(contour[j][0][0] - contour[i][0][0]) + 5
        diff_y = round(contour[j][0][1] - contour[i][0][1]) + 5
        classification = classification_matrix[diff_y][diff_x]
        classfied_fragment = ClassifiedContourFragment(classification,
                                                       (contour[i][0][0], contour[i][0][1]),
                                                       (contour[j][0][0], contour[j][0][1]))
        classified_fragments.append(classfied_fragment)

        classified_contours.append(classified_fragments)
    return classified_contours


def find_line_fragments(classified_contours, min_fragments=3, max_gap=2):
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


def create_allowed_direction_image(classified_contours, minimal_size, max_gap, allowed_directions, shape):
    line_fragments = find_line_fragments(classified_contours, minimal_size, max_gap)
    tmp_image = np.zeros(shape, np.uint8)
    for line in line_fragments:
        if line.classification in allowed_directions:
            for fragment in line.fragments:
                cv2.line(tmp_image, fragment.start_point, fragment.end_point, (255, 255, 255), 1)
    return tmp_image
