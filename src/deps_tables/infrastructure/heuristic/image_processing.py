from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from deps_tables.infrastructure.heuristic import utils as util

from .types import Bbox


def image_processing(img_vector: np.ndarray) -> List[Tuple[float]]:
    img, threshold = adaptive_threshold(img_vector, process_background=False, blocksize=15, c=-2)
    dmask1, lines_h = find_lines(threshold, regions=None, direction="horizontal", line_scale=15, iterations=0)
    return lines_h


def find_lines(threshold: float, regions=None, direction="horizontal", line_scale=15, iterations=0):  # noqa: WPS210
    """Finds horizontal and vertical lines by applying morphological
    transformations on an image.
    Parameters
    ----------
    threshold : object

        numpy.ndarray representing the thresholded image.

    regions : list, optional (default: None)

        List of page regions that may contain tables of the form x1,y1,x2,y2
        where (x1, y1) -> left-top and (x2, y2) -> right-bottom
        in image coordinate space.

    direction : string, optional (default: 'horizontal')

        Specifies whether to find vertical or horizontal lines.

    line_scale : int, optional (default: 15)

        Factor by which the page dimensions will be divided to get
        smallest length of lines that should be detected.
        The larger this value, smaller the detected lines. Making it
        too large will lead to text being detected as lines.

    iterations : int, optional (default: 0)

        Number of times for erosion/dilation is applied.
        For more information, refer `OpenCV's dilate <https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#dilate>`_.

    Returns
    -------
    dmask : object

        numpy.ndarray representing pixels where vertical/horizontal
        lines lie.

    lines : list

        List of tuples representing vertical/horizontal lines with
        coordinates relative to a left-top origin in
        image coordinate space.
    """
    lines = []

    if direction == "vertical":
        size = threshold.shape[0] // line_scale
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    elif direction == "horizontal":
        size = threshold.shape[1] // line_scale
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
    elif direction is None:
        raise ValueError("Specify direction as either 'vertical' or 'horizontal'")

    if regions is not None:
        region_mask = np.zeros(threshold.shape)
        for region in regions:
            x, y, w, h = region
            region_mask[y : y + h, x : x + w] = 1
        threshold = np.multiply(threshold, region_mask)

    threshold = cv2.erode(threshold, el)
    threshold = cv2.dilate(threshold, el)
    dmask = cv2.dilate(threshold, el, iterations=iterations)

    try:
        _, contours, _ = cv2.findContours(threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        # for opencv backward compatibility
        contours, _ = cv2.findContours(threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x1, x2 = x, x + w
        y1, y2 = y, y + h
        if direction == "vertical":
            lines.append(((x1 + x2) // 2, y2, (x1 + x2) // 2, y1))
        elif direction == "horizontal":
            lines.append((x1, (y1 + y2) // 2, x2, (y1 + y2) // 2))

    return dmask, lines


def adaptive_threshold(img: np.ndarray, process_background=False, blocksize=15, c=-2):
    """Thresholds an image using OpenCV's adaptiveThreshold.
    Parameters
    ----------
    imagename : string

        Path to image file.

    process_background : bool, optional (default: False)

        Whether or not to process lines that are in background.

    blocksize : int, optional (default: 15)

        Size of a pixel neighborhood that is used to calculate a
        threshold value for the pixel: 3, 5, 7, and so on.
        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.

    c : int, optional (default: -2)

        Constant subtracted from the mean or weighted mean.
        Normally, it is positive but may be zero or negative as well.
        For more information, refer `OpenCV's adaptiveThreshold
        <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.

    Returns
    -------
    img : object

        numpy.ndarray representing the original image.

    threshold : object

        numpy.ndarray representing the thresholded image.
    """

    if process_background:
        threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c)
    else:
        threshold = cv2.adaptiveThreshold(
            np.invert(img),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blocksize,
            c,
        )
    return img, threshold


def detect_lines_with_cv2(
    filtered_y_coords: List,
    table: Bbox,
    cells: List[Bbox],
    ocr: List[Dict[str, Any]],
) -> List[List[float]]:
    filtered_y_coords = sorted(filtered_y_coords)
    row_intervals = []
    min_hight = min([pred_cell[3] - pred_cell[1] for pred_cell in cells])

    for ind, row in enumerate(filtered_y_coords):
        if row < 0:
            continue
        if ind == len(filtered_y_coords) - 1:
            break
        if filtered_y_coords[ind + 1] - row >= min_hight - 5:
            row_intervals.append([row, filtered_y_coords[ind + 1]])

    last_line = False
    first_line = False
    for raw_cell in ocr:
        bbox = raw_cell["bbox"]
        cell_box = [bbox[0], bbox[1], bbox[2], bbox[3]]
        cell = []
        if util.check_any_intersection(cell_box, table):
            x1 = max(0, cell_box[0] - table[0])
            y1 = max(0, cell_box[1] - table[1])
            cell = [x1, y1, cell_box[2] - table[0], cell_box[3] - table[1]]
        else:
            continue

        x2 = table[2] - table[0]
        y2 = table[3] - table[1]
        if (
            util.is_box_valid(cell)
            and util.is_box_valid([0, filtered_y_coords[-1], x2, y2])
            and util.is_intersected_by_box(cell, [0, filtered_y_coords[-1], x2, y2], 0.1)
        ):
            last_line = True

        if (
            util.is_box_valid(cell)
            and util.is_box_valid([0, 0, x2, filtered_y_coords[0]])
            and util.check_any_intersection(cell, [0, 0, x2, filtered_y_coords[0]])
        ):
            first_line = True

    if last_line and table[3] - filtered_y_coords[-1] >= min_hight - 5:
        row_intervals.append([filtered_y_coords[-1], table[3]])
    if first_line and filtered_y_coords[0] >= min_hight - 5:
        row_intervals.insert(0, [0, filtered_y_coords[0]])
    return row_intervals
