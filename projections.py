import cv2
import numpy as np


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


"""Vertical projection of image"""


def vertical_projection(img, flip=False, is_binary = True):
    if flip:
        image = ~img
    else:
        image = np.copy(img)
    if is_binary:
        image = image / 255
    data = np.sum(image, 0, np.uint32)
    return data


"""Horizontal projection of image"""


def horizontal_projection(img, flip=False, is_binary = True):
    if flip:
        image = ~img
    else:
        image = np.copy(img)
    if is_binary:
        image = image / 255
    data = np.sum(image, 1, np.uint32)
    return data


"""Create graph of horizontal projection"""


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


"""Create graph of vertical projection"""


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


"""Find continous non zero value areas in data"""


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


