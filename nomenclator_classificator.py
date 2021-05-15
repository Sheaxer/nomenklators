import numpy as np
import cv2

from hough_line_transform_mapping_line_detection import handwritten_lines_detection
from image_preprocessor import binarize_image, get_vertical_contours_gabor
from lcm import contour_classification, find_line_fragments
from projections import horizontal_projection, find_areas, vertical_projection, vertical_projection_graph

"""Trying to detect vertical lines between 2 horizontal ones"""


def find_vertical_lines_between_horizontal_lines(equalized_image, horizontal_lines_image, horizontal_line_area_1,
                                                 horizontal_line_area_2, start_x, is_debug=False):
    # print("test")

    y_margin = 10
    x_margin = 50
    # start_x = horizontal_line_area_1[3]

    end_x = horizontal_line_area_1[4] + x_margin
    if end_x > equalized_image.shape[1]:
        end_x = equalized_image.shape[1]
    # try to look up and down from the lines - vertical lines usually go up and down
    start_row = horizontal_line_area_1[0] - y_margin
    start_row = 0 if start_row < 0 else start_row
    end_row = horizontal_line_area_2[1] + y_margin + 1
    end_row = equalized_image.shape[0] if end_row > equalized_image.shape[0] else end_row

    # use gabor filter on equalized image and get vertical contoures from it
    tmp_image = equalized_image[start_row: end_row, start_x: end_x + 1]
    cont_v = get_vertical_contours_gabor(tmp_image)

    # going to do some filtering
    filtered_contours = []

    # filter out contours that are too small
    for cont in cont_v:
        if cont.shape[0] >= int(horizontal_line_area_2[0] - horizontal_line_area_1[1]) - \
                int((horizontal_line_area_2[0] - horizontal_line_area_1[1]) / 20):
            filtered_contours.append(cont)

    # use lcm to classify contours so we only have one edge not two, clears up the space between lines
    classified_contours_v = contour_classification(filtered_contours)
    classified_lines_v = find_line_fragments(classified_contours_v, 1, 0)

    # drawing only one direction1
    contoured_vertical_image = np.zeros_like(tmp_image)
    for c_line in classified_lines_v:
        if c_line.classification == 3:
            for fragment in c_line.fragments:
                cv2.line(contoured_vertical_image, fragment.start_point, fragment.end_point, (255, 255, 255), 1)

    # debug
    # cv2.imwrite("export/cont.jpg",contoured_vertical_image)
    # now we procees with vertical projection to detect where the vertical lines are
    vert_proj = vertical_projection(contoured_vertical_image)
    vert_areas = find_areas(vert_proj, 10, 1)
    if vert_areas.shape[0] == 0:
        return None, contoured_vertical_image

    # debug
    """
    contoured_image_2_debug = cv2.cvtColor(contoured_vertical_image, cv2.COLOR_GRAY2BGR)
    vert_graph = vertical_projection_graph(vert_proj, contoured_vertical_image.shape)
    vert_graph = cv2.cvtColor(vert_graph, cv2.COLOR_GRAY2BGR)
    for area in vert_areas:
        cv2.line(contoured_image_2_debug, (area[0],0), (area[0],contoured_image_2_debug.shape[0]-1), (255,0,0),2)
        cv2.line(contoured_image_2_debug, (area[1],0), (area[1], contoured_image_2_debug.shape[0]-1), (0,255,0),2)

        cv2.line(vert_graph, (area[0],0), (area[0], vert_graph.shape[0]-1), (255,0,0),2)
        cv2.line(vert_graph, (area[1],0), (area[1], vert_graph.shape[0]-1), (0,255,0),2)

    cv2.imwrite("export/cont2.jpg", contoured_image_2_debug)
    cv2.imwrite("export/cont-vert-graph.jpg",vert_graph)

    vert_graph = vertical_projection_graph(vert_proj, contoured_vertical_image.shape)
    vert_graph = cv2.cvtColor(vert_graph, cv2.COLOR_GRAY2BGR)
    """

    # debug

    vert_areas_tmp = []
    for area in vert_areas:
        # apply horizontal projection to find height of potential line
        tmp_horizontal_proj = horizontal_projection(contoured_vertical_image[:, area[0]: area[1] + 1])

        # debug
        # cv2.imwrite("export/vert-line-horizontal.jpg",contoured_vertical_image[:, area[0]: area[1] + 1])
        # find non zero
        tmp_horizontal_areas = find_areas(tmp_horizontal_proj, 1, 12)

        if tmp_horizontal_areas.shape[0] == 0:
            continue
        else:
            # we found either a line or a fragment
            if tmp_horizontal_areas.shape[0] == 1:
                horizontal_area_length = tmp_horizontal_areas[0, 2]
            else:
                # should this be here?
                # merge smaller segments into one large one?
                horizontal_area_length = int(tmp_horizontal_areas[-1][1] - tmp_horizontal_areas[0][0])
            # if horizontal_area_length >= int(self.horizontal_lines[overlap[0]][0] - self.horizontal_lines[i][1]):
            vert_areas_tmp.append(np.append(area, [horizontal_area_length]))

    vert_areas = np.array(vert_areas_tmp)
    vert_areas_tmp = []

    for vert_area in vert_areas:
        # now we are filtering out and checking if there actually are horizontal lines here
        # margin, just to be safe
        vert_area_x_2 = vert_area[1] + 30

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


def classify_table(table, horizontal_lines, gray, image):
    if table is None:
        return None
    """Why is this super white"""
    # cv2.imwrite("export/a.jpg", gray)
    binarized = binarize_image(gray)
    # cv2.imwrite("export/b.jpg",binarized)

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
    skipped = False
    header_bodies = []
    contents = []

    if len(table) > 1:
        if len(table[0]) < 8:
            i = 1
            # skip wrong header
    while header_body_combo and i + 1 < len(table):
        if len(table[i]) == 0:
            contents.append(i)
            i += 1
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

    cipher_types = {"subs": False, "bigrams": False, "nulls": False, "codeBook": False, "inverse": False}
    # now try to see if header bodies are the same vs content are the same

    same_header_body = []
    same_header_body_line = [0]
    if len(header_bodies) > 0:
        for i in range(0, len(header_bodies) - 1):
            lower = table_means[header_bodies[i]][0] - (table_means[header_bodies[i]][0] * delta)
            upper = table_means[header_bodies[i]][0] + (table_means[header_bodies[i]][0] * delta)
            if lower < table_means[header_bodies[i + 1]][0] < upper:
                same_header_body_line.append(i + 1)
            else:
                same_header_body.append(same_header_body_line)
                same_header_body_line = [i + 1]

        same_header_body.append(same_header_body_line)

        lines_classified = []
        is_multiple = False
        if len(same_header_body) != 0:
            for header_body_line in same_header_body:
                column_count = 0
                for header_body in header_body_line:
                    column_count += len(table[header_bodies[header_body]])
                    is_multiple = True if (table_means[header_bodies[header_body]][1] * 2) < (table_means[header_bodies[header_body] + 1][1]) else is_multiple
                if column_count <= 6:
                    class_string = "0/{-1}"
                elif column_count < 20:
                    if cipher_types['subs'] is False:
                        cipher_types['subs'] = True
                        class_string = "1"
                    else:
                        class_string = "2"

                elif column_count <= 29:
                    if cipher_types['subs'] is False:
                        cipher_types['subs'] = True
                        class_string = "1"
                    elif cipher_types['bigrams'] is False:
                        cipher_types['bigrams'] = True
                        class_string = "2"
                    else:
                        class_string = "2"
                        # go back
                    if is_multiple:
                        class_string += "f"
                else:
                    cipher_types['subs'] = True
                    cipher_types['bigrams'] = True
                    class_string = "12"
                    if is_multiple:
                        class_string += "f"

                is_multiple = False
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

    if len(contents) > 0:
        if len(contents) > 1:
            for i in range(0, len(contents)):
                table_index = contents[i]
                if len(table[table_index]) > 6:
                    content_classification.append(
                        [
                            "1/2",
                            contents[i]
                        ]
                    )
                else:
                    class_string = ""
                    h = int(gray.shape[1] / 2)

                    sliced = binarized[horizontal_lines[contents[i]][1]: horizontal_lines[contents[i] + 1][0],
                             100:h - 100]
                    #cv2.imwrite("export/sliced2.jpg", sliced)

                    detected_lines, _ = handwritten_lines_detection(sliced)
                    line_number = len(detected_lines)

                    lined_image = np.copy(image)
                    """
                    for h_line in detected_lines:
                        cv2.line(lined_image, (100 + h_line[0], horizontal_lines[contents[i]][1] + h_line[1]),
                                 (100 + h_line[2], horizontal_lines[contents[i]][1] + h_line[3]),
                                 (255, 0, 0), 2)
                    cv2.imwrite("export/lined-image.jpg", lined_image)
                    """
                    if line_number < 6:
                        if cipher_types['subs'] is False:
                            if i > 0:
                                cipher_types['subs'] = True
                            class_string = "1"
                        else:
                            class_string = "0/{-1}"

                        content_classification.append([
                            class_string,
                            contents[i]])
                    else:
                        class_string = "V/{-1}"
                        content_classification.append([
                            class_string,
                            contents[i]])

        else:
            sliced = binarized[horizontal_lines[contents[0]][0]: horizontal_lines[contents[0] + 1][1],
                     100:int(gray.shape[1] / 2) - 100]

            #cv2.imwrite("export/sliced3.jpg", sliced)

            detected_lines, _ = handwritten_lines_detection(sliced)
            line_number = len(detected_lines)
            """
            print(str(line_number))
            lined_image = np.copy(image)
            for h_line in detected_lines:
                cv2.line(lined_image, (h_line[0], horizontal_lines[contents[0]][1] + h_line[1]),
                         (h_line[2], horizontal_lines[contents[0]][1] + h_line[3]),
                         (255, 0, 0), 2)
            cv2.imwrite("export/lined-image2.jpg", lined_image)
            """
            if line_number > 7:
                class_string = "V/{-1}"
            else:
                class_string = "0/{-1}"
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
    if binarized.shape[0] - start_y > 400:
        # cv2.imwrite("export/binarized.jpg",binarized)

        sliced = binarized[start_y + 10:, 300:int(gray.shape[1] / 2) - 100]
        horizont_proj = horizontal_projection(binarized)
        horizont_areas = find_areas(horizont_proj, 1, 2)

        vert_proj = vertical_projection(binarized)
        vert_areas = find_areas(vert_proj, 2, 80)

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

            detected_lines, _ = handwritten_lines_detection(sliced)
            """
            cv2.imwrite("export/sliced4.jpg", sliced)

            lined_image = np.copy(image)
            for h_line in detected_lines:
                cv2.line(lined_image, (h_line[0], start_y + 10 + h_line[1]), (h_line[2], start_y + 10 + h_line[3]),
                         (255, 0, 0), 2)

            cv2.imwrite("export/lined-image3.jpg", lined_image)
            """
            line_number = len(detected_lines)
            #print(str(line_number))

            if line_number > 7:
                class_string = "V/{-1}"
            else:
                class_string = "0/{-1}"
            extra_area_classification = [
                class_string,
                (start_x, start_y + start_y2),
                (end_x, start_y + + start_y2),
                (start_x, start_y + end_y2),
                (end_x, start_y + end_y2)
            ]

    # create string
    used_colors = []
    classification_string = ""

    if len(table_classification) > 0:
        for table_class in table_classification:
            classification_string += table_class[0]
            classification_string += "\n"

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

                classification_string += table_class_line[0]
                classification_string += " "
                classification_string += str(table[table_class_line[1]][0][0])
                classification_string += " "
                classification_string += str(table[table_class_line[1]][-1][1])
                classification_string += " "
                classification_string += str(table[table_class_line[1]][0][2])
                classification_string += " "
                classification_string += str(table[table_class_line[1]][-1][3])
                classification_string += "\n"
            classification_string += "\n"

            color = list(np.random.random(size=3) * 256)
            while color in used_colors:
                color = list(np.random.random(size=3) * 256)
            used_colors.append(color)
            cv2.line(image, (start_x, start_y), (end_x, start_y), color, 4)
            cv2.line(image, (start_x, start_y), (start_x, end_y), color, 4)
            cv2.line(image, (end_x, start_y), (end_x, end_y), color, 4)
            cv2.line(image, (start_x, end_y), (end_x, end_y), color, 4)

    if len(content_classification) > 0:
        for cont in content_classification:
            classification_string += cont[0]
            classification_string += "\n"
            start_y = horizontal_lines[cont[1]][0]
            end_y = horizontal_lines[cont[1] + 1][1]
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

            cv2.line(image, (start_x, start_y), (end_x, start_y), color, 4)
            cv2.line(image, (start_x, start_y), (start_x, end_y), color, 4)
            cv2.line(image, (end_x, start_y), (end_x, end_y), color, 4)
            cv2.line(image, (start_x, end_y), (end_x, end_y), color, 4)

            classification_string += "c "
            classification_string += str((start_x, start_y))
            classification_string += " "
            classification_string += str((end_x, start_y))
            classification_string += " "
            classification_string += str((start_x, end_y))
            classification_string += " "
            classification_string += str((end_x, end_y))
            classification_string += "\n"
            classification_string += "\n"
        classification_string += "\n"

    if extra_area_classification is not None:

        color = list(np.random.random(size=3) * 256)
        while color in used_colors:
            color = list(np.random.random(size=3) * 256)
        used_colors.append(color)

        cv2.line(image, extra_area_classification[1], extra_area_classification[2], color, 4)
        cv2.line(image, extra_area_classification[1], extra_area_classification[3], color, 4)
        cv2.line(image, extra_area_classification[2], extra_area_classification[4], color, 4)
        cv2.line(image, extra_area_classification[3], extra_area_classification[4], color, 4)

        classification_string += extra_area_classification[0]
        classification_string += "\n"
        classification_string += "c "
        classification_string += str(extra_area_classification[1])
        classification_string += " "
        classification_string += str(extra_area_classification[2])
        classification_string += " "
        classification_string += str(extra_area_classification[3])
        classification_string += " "
        classification_string += str(extra_area_classification[4])
        classification_string += "\n"

    return classification_string, image


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
                    elif table[i][j][4] > upper:
                        if table[i][j][1][0] - (table[i][j][0][0] + width_mean) > lower:
                            table_line_tmp = table[i][0:j]
                            table_line_tmp.append(
                                [
                                    table[i][j][0],
                                    (int(table[i][j][0][0] + width_mean), table[i][j][0][1]),
                                    table[i][j][2],
                                    (int(table[i][j][2][0] + width_mean), table[i][j][3][1]),
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

    # try to see if we need to add left or right column?
    return table
