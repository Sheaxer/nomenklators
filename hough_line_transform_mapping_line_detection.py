import numpy as np
import cv2
from sympy import Point, Polygon, Line
import math
from skimage.transform import hough_line


"""Transformation line back to euclid system"""

def findLine(rho, theta):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * (a))
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * (a))

    return x1, y1, x2, y2

"""Divide connectedComponents into 3 subsets

1st are probably normal numbers 0.5* AH < H <3*AH and 0.5 * AW <= W
"""
def divide(centroids, stats, ah, aw):
    centroids1 = []
    centroids2 = []
    centroids3 = []

    stats1 = []
    stats2 = []
    stats3 = []

    for i in range(len(centroids)):
        centroid = centroids[i]
        stat = stats[i]
        h = stat[cv2.CC_STAT_HEIGHT]
        w = stat[cv2.CC_STAT_WIDTH]

        # if 0.5*AH <= H < 3*AH
        # regular ccs -
        if ((0.5*ah) <= h < (3 * ah)) and ((0.5 * aw) <= w):
            centroids1.append(centroid)
            stats1.append(stat)
        # subset 2 - all large ccs - capital letter etc
        elif h >= 3 * ah:
            centroids2.append(centroid)
            stats2.append(stat)

        # subset 3 - accents, punctuation marks, small characters
        elif (h < (3 * ah) and 0.5 * aw > w) or ((h < 0.5 * ah) and (0.5 * aw < w)):
            centroids3.append(centroid)
            stats3.append(stat)

    return (centroids1, stats1), (centroids2, stats2), (centroids3, stats3)


def findComponents(image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    avg_height = 0
    avg_width = 0
    for stat in stats:
        #calculate average height of connected component / character
        avg_height += stat[cv2.CC_STAT_HEIGHT]
        avg_width += stat[cv2.CC_STAT_WIDTH]
    avg_height /= num_labels
    avg_width /= num_labels
    return (labels, avg_height, avg_width,centroids, stats)

"""Partition the connected component into equaly sized blocks to have more representative points for Hough"""

def partitionCC(stats, aw, image_width):
    centroids = []
    mapP = []
    for pos in range(len(stats)):
        stat = stats[pos]
        l = stat[cv2.CC_STAT_LEFT]
        t = stat[cv2.CC_STAT_TOP]
        h = stat[cv2.CC_STAT_HEIGHT]
        w = stat[cv2.CC_STAT_WIDTH]
        top_range = int(math.ceil(w / aw))
        for i in range(1, top_range + 1):
            start = (i - 1) * aw
            end = i * aw if i * aw < w else w


            # create bouding box of cc
            # polyCen = Polygon ((l, t), (l + end, t), (l + end, t + h), (l, t + h)).centroid;
            polyCen = Polygon((l + start, t), (l + start + end, t), (l + start + end, t + h),
                              (l + start, t + h)).centroid
            centroids.append((polyCen.x, polyCen.y))
            mapP.append(pos)
    return centroids, mapP

"""Put Centroids into map"""

def showCentroids(image, centroids):
    demo = np.zeros((np.shape(image)), dtype=np.uint8)

    if centroids is not None:
        for centroid in centroids:
            try:
                demo[int(centroid[1]), int(centroid[0])] = 255
            except: IndexError


    return demo

#meat of the method
def find_ccs_on_line(acc, thetas, dists, centroids, pos, dominant_skew = None,n1=5, n2=9):
    max_cell = np.unravel_index(acc.argmax(), acc.shape)
    if acc[max_cell[0]][max_cell[1]] < n1:
        return None, None, None, False

    p_index = max_cell[0]
    th_index = max_cell[1]

    append_line = True
    # print(acc[p_index,th_index])
    p = dists[p_index]
    th = thetas[th_index]
    if dominant_skew is not None:
        if acc[max_cell[0], max_cell[1]] < n2:
            if th < (dominant_skew - math.radians(2)) or th > (dominant_skew + math.radians(2)):
                append_line = False
    p_min = p - 5
    p_max = p + 5
    old_position = pos[0]
    number_of_pts = 0
    total_pts = 0
    voting_ccs = []
    y_cos = np.cos(th)
    y_sin = np.sin(th)
    voting_pts = []
    for i in range(0, len(pos)):
        if pos[i] != old_position:
            voting_ccs.append([number_of_pts, total_pts, voting_pts, old_position])
            total_pts = 1
            old_position = pos[i]
            number_of_pts = 0
            voting_pts = []
        else:
            total_pts += 1
        x = centroids[i][0]
        y = centroids[i][1]
        centroid_p = x * y_cos + y * y_sin
        voting_pts.append(i)
        if p_min <= centroid_p <= p_max:
            number_of_pts += 1
    return voting_ccs, p, th, append_line


def detect_if_same_text_line(lines, image):
    (h, w) = image.shape
    sympy_lines = []
    for i in range(0, len(lines)):
        x1, y1, x2, y2 = findLine(lines[i][0], lines[i][1])
        sympy_lines.append(Line((x1, y1), (x2, y2)))

    distances = []
    adjacency = []
    for i in range(0, len(lines) - 1):
        for j in range(i + 1, len(lines)):
            inter_p = sympy_lines[i].intersection(sympy_lines[j])
            if len(inter_p) > 0:
                if 0 < inter_p[0].x < w:
                    distance = (int(inter_p[0].distance(Point(w/2,inter_p[0].y))))
                    adjacency.append([i,j,distance])
                    distances.append(distance)

    distances = np.array(distances)
    merge_indices = []

    if distances.shape[0] > 0:
        distance_mean = np.average(distances)

        for adj in adjacency:
            if adj[2] < distance_mean:
                if len(merge_indices) == 0:
                    merge_indices.append([adj[0],adj[1]])
                else:
                    found = False
                    for merge in merge_indices:
                        if adj[0] in merge:
                            merge.append(adj[1])
                            found = True
                        elif adj[1] in merge:
                            merge.append(adj[0])
                            found = True
                    if not found:
                        merge_indices.append([adj[0],adj[1]])


    #print("done")
    return merge_indices


def isAdjacent(line_i, line_j):
    start_line = Line((0, 0), (0, 100))
    (_, yi) = line_i.intersection(start_line)[0]
    (_, yj) = line_j.intersection(start_line)[0]

    if abs(yi - yj) < 100:
        return True
    else:
        return False


def handwritten_lines_detection(image, is_thresholded=True):
    if is_thresholded is False:
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        thresh = image
    image_width = image.shape[1]
    (labels, avg_height, avg_width, centroids, stats) = findComponents(thresh)
    ((centroids1, stats1), (centroids2, stats2), (centroids3, stats3)) = divide(centroids, stats, avg_height, avg_width)
    centroidP, mapP = partitionCC(stats1, int(avg_height), image_width)
    # orig_centroidP = np.copy(centroidP)
    # orig_mapP = mapP.copy()
    centroidImage = showCentroids(thresh, centroidP)
    # cv2.imwrite("export/centroid.jpg",centroidImage)
    # lines = None
    line_angles = np.zeros(10)
    tested_angle = np.linspace(math.radians(85), math.radians(95), 10, endpoint=True)
    acc, thetas, dists = hough_line(centroidImage, theta=tested_angle)

    detected_lines = []
    voting, lineP, lineTh, append_line = find_ccs_on_line(acc, thetas, dists, centroidP, mapP)
    if lineTh is not None:
        j=0
        while tested_angle[j] < lineTh:
            j+=1
        line_angles[j]+=1
    #try finding dominant skew angle?

    while voting is not None:
        removed_indexes = []
        detected_ccs = []
        for v in voting:
            if v[0] > 0 and v[0] >= int(v[1] / 2):
                detected_ccs.append(v[3])
                removed_indexes.extend(v[2])
        sorted_indices_to_delete = sorted(removed_indexes, reverse=True)
        for index in sorted_indices_to_delete:
            del mapP[index]
            del centroidP[index]
        centroidImage = showCentroids(thresh, centroidP)
        acc, thetas, dists = hough_line(centroidImage, theta=tested_angle)
        if append_line:
            detected_lines.append([lineP, lineTh, detected_ccs])
        voting, lineP, lineTh, append_line = find_ccs_on_line(acc, thetas, dists, centroidP, mapP,
                                                              tested_angle[np.argmax(line_angles)])
        if append_line:
            j = 0
            while tested_angle[j] < lineTh:
                j += 1
            line_angles[j] += 1

    merge_indices = detect_if_same_text_line(detected_lines, thresh)
    transformed_detected_lines = []

    merged_detected_lines = []
    for merge in merge_indices:
        merged_detected_line= detected_lines[merge[0]]
        for i in range(1, len(merge)):
            merged_detected_line[2].extend(detected_lines[merge[i]][2])
    for i in range(0, len(detected_lines)):
        was_merged = False
        for merge in merge_indices:
            if i in merge:
                was_merged = True
        if not was_merged:
            merged_detected_lines.append(detected_lines[i])

    for detected_line in merged_detected_lines:
        x1, y1, x2, y2 = findLine(detected_line[0], detected_line[1])
        transformed_detected_lines.append([x1, y1, x2, y2])
    return transformed_detected_lines, merged_detected_lines

"""
if __name__ == "__main__":
    image = cv2.imread("hstam_4 d Nr 1218 Teil 1_0046.jpg")
    image2 = np.copy(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cannied = cv2.Canny(image_gray,150,250,None,3,True)
    cv2.imwrite("thresholded.jpg",thresholded)
    cv2.imwrite("cannied.jpg",cannied)

    contours, hierarchy = cv2.findContours(thresholded,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoured = np.zeros_like(thresholded)
    cv2.drawContours(contoured,contours,-1,(255,255,255))

    lines, _ = handwritten_lines_detection(thresholded)
    for line in lines:
        cv2.line(image2, (line[0],line[1]), (line[2],line[3]), (255,0,0),2)

    cv2.imwrite("lined-image.jpg",image2)
"""
