import logging
from collections import defaultdict as dd
from itertools import combinations
from operator import itemgetter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from deps_tables.domain.entities import (
    BoundingBoxEntity,
    CellCoordinates,
    CellEntity,
    ColumnEntity,
    RelativeAreaEntityWithPage,
    RowEntity,
    TableEntity,
    TextLineEntity,
)

from .types import Bbox, Cell, CellTuple, Interval, OCROutput, TableElements

logger = logging.getLogger(__name__)


def bounding_box_to_tuple(elements: List[BoundingBoxEntity]) -> List[CellTuple]:
    format_boxes = []
    for elem in elements:
        left = elem.left_top_point.x
        top = elem.left_top_point.y
        right = elem.right_bottom_point.x
        bottom = elem.right_bottom_point.y
        box = (max(left, 0), max(top, 0), max(right, 0), max(bottom, 0))
        if right > left and bottom > top:
            format_boxes.append(box)
    return format_boxes


def check_any_intersection(box1: Bbox, box2: Bbox) -> bool:
    rule1 = min(box1[2], box2[2]) > max(box1[0], box2[0])
    rule2 = min(box1[3], box2[3]) > max(box1[1], box2[1])
    return rule1 and rule2


def is_intersected_by_box(box1: Bbox, box2: Bbox, threshold: float) -> bool:
    intersection_area = get_intersection_area(box1, box2)
    intersection_ratio = intersection_area / min(calculate_bbox_area(box2), calculate_bbox_area(box1))
    return intersection_ratio >= threshold


def merge_bboxes(bboxes: List[Bbox], merge_threshold: float) -> List[Bbox]:
    tuple_bboxes = [tuple(bbox) for bbox in bboxes]
    updated_bboxes = set(tuple_bboxes)

    for bbox1, bbox2 in combinations(tuple_bboxes, 2):
        try:
            if is_intersected_by_box(bbox1, bbox2, merge_threshold):
                merged_bbox = (
                    min(bbox1[0], bbox2[0]),
                    min(bbox1[1], bbox2[1]),
                    max(bbox1[2], bbox2[2]),
                    max(bbox1[3], bbox2[3]),
                )

                if bbox1 in updated_bboxes and bbox2 in updated_bboxes:
                    updated_bboxes.remove(bbox1)
                    updated_bboxes.remove(bbox2)
                    updated_bboxes.add(merged_bbox)
        except ZeroDivisionError:
            logger.info("exception in merge_bboxes")

    return [list(bbox) for bbox in updated_bboxes]


def get_intersection_area(valid_bbox: Bbox, predict_bbox: Bbox) -> int:
    return calculate_bbox_area(
        [
            max(valid_bbox[0], predict_bbox[0]),
            max(valid_bbox[1], predict_bbox[1]),
            min(valid_bbox[2], predict_bbox[2]),
            min(valid_bbox[3], predict_bbox[3]),
        ],
    )


def calculate_bbox_area(bbox: Bbox) -> int:
    """
    Function calculates rectangle's square.
    """
    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    return width * height


def check_intersection_small_box_big_box(box1: Bbox, box2: Bbox, threshold=0.07) -> float:
    return check_any_intersection(box1, box2) and is_intersected_by_box(box1, box2, threshold)


def unite_cells(*args) -> CellTuple:
    x1 = min(args, key=itemgetter(0))[0]
    y1 = min(args, key=itemgetter(1))[1]
    x2 = max(args, key=itemgetter(2))[2]
    y2 = max(args, key=itemgetter(3))[3]
    return x1, y1, x2, y2


def merge_overlapping_boxes(boxes: List[CellTuple], threshold: float) -> List[CellTuple]:
    overlapping_boxes = []
    used = [False for _ in boxes]
    for ind1, box1 in enumerate(boxes):
        if used[ind1]:
            continue
        stack = [ind1]
        overlapping_box = box1
        while stack:
            box_idx = stack.pop()
            used[box_idx] = True
            for ind2, box2 in enumerate(boxes):
                if not used[ind2] and is_intersected_by_box(boxes[box_idx], box2, threshold):
                    overlapping_box = unite_cells(overlapping_box, box2)
                    stack.append(ind2)
        overlapping_boxes.append(overlapping_box)
    return overlapping_boxes


def get_intersected_box_index(cell: List, elements: List) -> List:
    intersected_elements = []
    for elem_ind, elem_box in enumerate(elements):
        min_box = min(calculate_bbox_area(elem_box), calculate_bbox_area(cell))
        if check_any_intersection(elem_box, cell) and min_box != 0:
            intersection_area = get_intersection_area(elem_box, cell)
            intersection_ratio = intersection_area / min_box
            intersected_elements.append((intersection_ratio, elem_ind))
    return intersected_elements


def reformat_ocr_output(ocr: List[TextLineEntity]) -> List[OCROutput]:
    output: List[OCROutput] = []
    for line in ocr:
        for word_box in line.word_boxes:
            left = word_box.bbox.left_top_point.x
            top = word_box.bbox.left_top_point.y
            right = word_box.bbox.right_bottom_point.x
            buttom = word_box.bbox.right_bottom_point.y
            output.append(
                {
                    "bbox": [left, top, right, buttom],
                    "content": word_box.content,
                    "line": line.id,
                },
            )
    return output


def cell_to_content(
    cells: List[CellTuple],
    ocr: List[Dict[str, Any]],
    table_top: int,
    table_left: int,
    threshold: float,
) -> Dict[tuple, Any]:
    cells_to_content: Dict[tuple, Any] = dd(dict)
    for cell in cells:
        x1 = cell[0] + table_left
        y1 = cell[1] + table_top
        x2 = cell[2] + table_left
        y2 = cell[3] + table_top
        abs_cell = [x1, y1, x2, y2]
        if ocr:
            cell_text = []
            boxes = []
            lines = []

            for element in ocr:
                box = element["bbox"]
                ocr_cell = [box[0], box[1], box[2], box[3]]
                inter_area = get_intersection_area(abs_cell, ocr_cell)
                if check_any_intersection(abs_cell, ocr_cell) and inter_area / calculate_bbox_area(ocr_cell) >= threshold:
                    cell_text.append(element["content"])
                    boxes.append(ocr_cell)
                    lines.append(element["line"])
            cells_to_content[tuple(cell)] = {
                "cell_content": cell_text,
                "cell_boxes": boxes,
                "line": lines,
            }
        else:
            cells_to_content[tuple(cell)] = {
                "cell_content": [],
                "cell_boxes": [],
                "abs_cell_coord": abs_cell,
                "cell_boxes_relative": [],
                "line": [],
            }
    return cells_to_content


def get_intersection_area_coord(valid_bbox: Bbox, predict_bbox: Bbox) -> Bbox:
    return [
        max(valid_bbox[0], predict_bbox[0]),
        max(valid_bbox[1], predict_bbox[1]),
        min(valid_bbox[2], predict_bbox[2]),
        min(valid_bbox[3], predict_bbox[3]),
    ]


def cut_intersection_by_y(bbox1: Bbox, bbox2: Bbox, intersec_area: Bbox):
    anti_overlap_number = 1
    top_box = bbox1 if bbox1[1] > bbox2[1] else bbox2
    bot_box = bbox1 if bbox1[1] < bbox2[1] else bbox2
    new_top_box = [top_box[0], intersec_area[3] + anti_overlap_number, top_box[2], top_box[3] - anti_overlap_number]
    new_bot_box = [bot_box[0], bot_box[1] + anti_overlap_number, bot_box[2], intersec_area[1] - anti_overlap_number]

    return top_box, bot_box, new_top_box, new_bot_box


def cut_intersection_by_x(bbox1: Bbox, bbox2: Bbox, intersec_area: Bbox):
    anti_overlap_number = 1
    right = bbox1 if bbox1[0] > bbox2[0] else bbox2
    left = bbox1 if bbox1[0] < bbox2[0] else bbox2

    new_right = [intersec_area[2] + anti_overlap_number, right[1], right[2] - anti_overlap_number, right[3]]
    new_left = [left[0] + anti_overlap_number, left[1], intersec_area[0] - anti_overlap_number, left[2]]

    return right, left, new_right, new_left


def column_neighbor(box1: Bbox, box2: Bbox) -> bool:
    box1_x_centre = (box1[0] + box1[2]) / 2
    box2_x_centre = (box2[0] + box2[2]) / 2
    rule1 = box2[0] < box1_x_centre < box2[2]
    rule2 = box1[0] < box2_x_centre < box1[2]

    return rule1 or rule2


def row_neighbor(box1: Bbox, box2: Bbox) -> bool:
    box1_y_centre = (box1[1] + box1[3]) / 2
    box2_y_centre = (box2[1] + box2[3]) / 2
    rule1 = box2[1] < box1_y_centre < box2[3]
    rule2 = box1[1] < box2_y_centre < box1[3]
    return rule1 or rule2


def is_box_valid(box: Bbox) -> bool:
    return 0 <= box[0] < box[2] and 0 <= box[1] < box[3]


def crop_boxes(boxes: Sequence) -> Tuple[Dict[Tuple, Any], Dict[Tuple, Any]]:
    boxes_and_they_modifications_vertical = dd(list)
    boxes_and_they_modifications_horisontal = dd(list)
    for bbox1, bbox2 in combinations(boxes, 2):
        if isinstance(bbox1, dict):
            bbox1 = bbox1["bbox"]
            bbox2 = bbox2["bbox"]
        if check_any_intersection(bbox1, bbox2) and bbox1 != bbox2:
            intersec_area = get_intersection_area_coord(bbox1, bbox2)

            if column_neighbor(bbox1, bbox2):
                top_box, bot_box, new_top_box, new_bot_box = cut_intersection_by_y(bbox1, bbox2, intersec_area)
                boxes_and_they_modifications_vertical[tuple(top_box)].append(new_top_box)
                boxes_and_they_modifications_vertical[tuple(bot_box)].append(new_bot_box)

            if row_neighbor(bbox1, bbox2):
                right, left, new_right, new_left = cut_intersection_by_x(bbox1, bbox2, intersec_area)
                boxes_and_they_modifications_horisontal[tuple(right)].append(new_right)
                boxes_and_they_modifications_horisontal[tuple(left)].append(new_left)
    return boxes_and_they_modifications_vertical, boxes_and_they_modifications_horisontal


def resize_cells(predicted_cells: Sequence[Sequence[int]]) -> List[CellTuple]:
    normalized_cells = [tuple(box) for box in predicted_cells]
    boxes_and_they_modifications_vertical, boxes_and_they_modifications_horisontal = crop_boxes(normalized_cells)

    updated_cells = []
    for orig_box, modifications in boxes_and_they_modifications_vertical.items():
        if modifications:
            new_box = modifications[-1]
            if boxes_and_they_modifications_horisontal[orig_box]:
                addit_box = boxes_and_they_modifications_horisontal[orig_box][-1]
                new_box = [addit_box[0], new_box[1], addit_box[2], new_box[3]]
            if is_box_valid(new_box):
                updated_cells.append(new_box)
        elif is_box_valid(orig_box):
            updated_cells.append(orig_box)
    new_cells = []
    for c in normalized_cells:
        if c not in list(boxes_and_they_modifications_vertical.keys()):
            new_cells.append((c[0], c[1], c[2], c[3]))
    updated_cells.extend(new_cells)
    return updated_cells


def resize_ocr_cells(ocr_dict: Sequence[OCROutput]) -> List[Dict[str, Any]]:
    new_ocr_dict = []
    ocr_cells = {}
    for ocr_box in ocr_dict:
        ocr_cells[tuple(ocr_box["bbox"])] = {
            "content": ocr_box["content"],
            "line": ocr_box["line"],
        }

    boxes_and_they_modifications_vertical, boxes_and_they_modifications_horisontal = crop_boxes(ocr_dict)
    for orig_box, modifications in boxes_and_they_modifications_vertical.items():
        if modifications:
            new_box = modifications[-1]
            if boxes_and_they_modifications_horisontal[orig_box]:
                addit_box = boxes_and_they_modifications_horisontal[orig_box][-1]
                new_box = [addit_box[0], new_box[1], addit_box[2], new_box[3]]
                if is_box_valid(new_box):
                    new_ocr_dict.append(
                        {
                            "bbox": new_box,
                            "content": ocr_cells[orig_box]["content"],
                            "line": ocr_cells[orig_box]["line"],
                        },
                    )
                elif is_box_valid(orig_box):
                    new_ocr_dict.append(
                        {
                            "bbox": orig_box,
                            "content": ocr_cells[orig_box]["content"],
                            "line": ocr_cells[orig_box]["line"],
                        },
                    )
                else:
                    logger.info("couldn't append resized cell (bad coordinates)")
                    continue

            elif is_box_valid(new_box):
                new_ocr_dict.append(
                    {
                        "bbox": new_box,
                        "content": ocr_cells[orig_box]["content"],
                        "line": ocr_cells[orig_box]["line"],
                    },
                )
        elif is_box_valid(orig_box):
            new_ocr_dict.append(
                {
                    "bbox": orig_box,
                    "content": ocr_cells[orig_box]["content"],
                    "line": ocr_cells[orig_box]["line"],
                },
            )

    for box in ocr_cells.keys():
        if box not in list(boxes_and_they_modifications_vertical.keys()) and is_box_valid(box):
            new_ocr_dict.append({"bbox": box, "content": ocr_cells[box]["content"], "line": ocr_cells[box]["line"]})
    return new_ocr_dict


def create_table_from_lines(columns: Sequence[Sequence[int]], rows: Sequence[Sequence[int]]):
    cells_list = {}
    columns_intervals = sorted(columns, key=lambda x: (x[0], x[1]))
    rows_intervals = sorted(rows, key=lambda x: x[0])

    for col_ind, colunm in enumerate(columns_intervals):
        for row_ind, row in enumerate(rows_intervals):
            if row_ind in {0, len(rows_intervals) - 1}:
                key1 = (colunm[0] + 1, row[0], colunm[2] - 1, row[1])
                col_row_info = {"col": col_ind, "row": row_ind}
                cells_list.update({key1: col_row_info})
                continue

            if row[1] <= colunm[3] and row[0] >= colunm[1]:
                key2 = (colunm[0] + 1, row[0], colunm[2] - 1, row[1])
                col_row_info2 = {"col": col_ind, "row": row_ind}
                cells_list.update({key2: col_row_info2})

    return cells_list


def sort_text_in_the_cell(cells_to_content: Dict, cell: Bbox, is_pdf_searchble: bool) -> str:
    text = ""
    cell_content = cells_to_content[tuple(cell)]["cell_content"]
    if cell_content:
        boxes = cells_to_content[tuple(cell)]["cell_boxes"]
        lines = cells_to_content[tuple(cell)]["line"]
        for_sort = dd(list)
        for con, box, text_line in zip(cell_content, boxes, lines):
            for_sort[text_line].append([con, box])

        for line in sorted(for_sort.keys(), reverse=is_pdf_searchble):
            content_box = for_sort[line]
            sorted_content_box = sorted(content_box, key=lambda x: x[1][0])
            text += " ".join([elem[0] for elem in sorted_content_box])
    return text


def build_relative_table(table_elements: TableElements, image_shape: Tuple[int, int]) -> TableEntity:
    table_box: List[int] = table_elements.table
    cells_list: List[Cell] = table_elements.cells
    h, w = image_shape
    t_width = table_box[2] - table_box[0]
    t_height = table_box[3] - table_box[1]
    columns_list = [ColumnEntity(x=col[0] / t_width) for col in table_elements.columns]
    rows_list = [RowEntity(y=row[0] / t_height) for row in table_elements.rows]
    area = RelativeAreaEntityWithPage(
        left=table_box[0] / w,
        top=table_box[1] / h,
        width=t_width / w,
        height=t_height / h,
    )
    cells = [
        CellEntity(
            value=cell.value,
            coordinates=CellCoordinates(
                column=cell.column,
                row=cell.row,
                column_span=1,
                row_span=1,
            ),
        )
        for cell in cells_list
    ]
    return TableEntity(cells=cells, columns=columns_list, rows=rows_list, coordinates=area)


class LineDetector:
    AREA_THRESHOLD = 0.2
    CELL_THRESHOLD = 0.3
    HEURISTIC_THRESHOLD = 0.3

    def __init__(self, table_elements: TableElements, ocr: List, is_pdf_searchble: bool):
        self._table_elements: TableElements = table_elements
        self._ocr = ocr
        self._is_pdf_searchble = is_pdf_searchble

    def detect_lines(self) -> Tuple[List[Cell], List[Interval]]:
        cells_to_content = cell_to_content(
            self._table_elements.cells,
            self._ocr,
            self._table_elements.table[1],
            self._table_elements.table[0],
            self.CELL_THRESHOLD,
        )
        indexing_list = self._create_list(cells_to_content)
        heuristic_detector = HeuristicColumnDetector(indexing_list, self._table_elements.columns, self.HEURISTIC_THRESHOLD)
        self._table_elements.columns = heuristic_detector.detect_columns()
        indexing_list = self._create_list(cells_to_content)
        indexing_list, undetected_cells_list = self._merge_multiple_cells(indexing_list)
        undetected_cells_list = self._generate_borders(undetected_cells_list)
        undetected_cells_list = [self._resize_new_cells(indexing_list, cell) for cell in undetected_cells_list]
        indexing_list += undetected_cells_list
        cells_to_content_new = cell_to_content(
            [cell[2] for cell in indexing_list],
            self._ocr,
            self._table_elements.table[1],
            self._table_elements.table[0],
            self.CELL_THRESHOLD,
        )

        cells_list = []
        for cell in indexing_list:
            cell_obj = Cell(
                column=cell[0],
                row=cell[1],
                value=sort_text_in_the_cell(cells_to_content_new, cell[2], self._is_pdf_searchble),
            )
            cells_list.append(cell_obj)
        cells_list = sorted(cells_list, key=lambda x: (x.row, x.column))
        return cells_list, self._table_elements.columns

    @staticmethod
    def _generate_new_cell(cell1: List, cell2: List) -> List:
        coords = (
            min(cell1[2][0], cell2[2][0]),
            min(cell1[2][1], cell2[2][1]),
            max(cell1[2][2], cell2[2][2]),
            max(cell1[2][3], cell2[2][3]),
        )
        return [cell1[0], cell1[1], coords]

    @staticmethod
    def _resize_new_cells(index_old: List, cell_new: List) -> List:
        for cell_old in index_old:
            if check_any_intersection(cell_new[2], cell_old[2]):
                if cell_old[0] < cell_new[0]:
                    cell_new[2] = cell_tuple_replace(cell_new[2], cell_old[2], 0, 2)
                elif cell_old[0] > cell_new[0]:
                    cell_new[2] = cell_tuple_replace(cell_new[2], cell_old[2], 2, 0)
                if cell_old[1] < cell_new[1]:
                    cell_new[2] = cell_tuple_replace(cell_new[2], cell_old[2], 1, 3)
                elif cell_old[1] > cell_new[1]:
                    cell_new[2] = cell_tuple_replace(cell_new[2], cell_old[2], 3, 1)
        return cell_new

    def _create_list(self, cells_to_content: Dict) -> List[List]:
        indexing_list = []
        for table_cell in self._table_elements.cells:
            intersected_rows, intersected_columns = self._find_row_column_index(table_cell)
            cell_row_ind = self._find_first_intersected_element(intersected_rows)
            cell_col_ind = self._find_first_intersected_element(intersected_columns)
            content = cells_to_content[tuple(table_cell)]["cell_content"]
            if cell_row_ind == -1:
                logger.info(f"no row for cell: {table_cell}, {content}")
                continue
            if cell_col_ind == -1:
                logger.info(f"no row for cell: {table_cell}, {content}")
                continue
            indexing_list.append([cell_col_ind, cell_row_ind, table_cell])
        return indexing_list

    def _find_row_column_index(self, cell: List) -> Tuple:
        h, w = self._table_elements.image.shape
        row_boxes = [[0, row[0], w - 1, row[1]] for row in self._table_elements.rows]
        column_boxes = [[column[0], 0, column[1], h - 1] for column in self._table_elements.columns]
        intersected_rows = get_intersected_box_index(cell, row_boxes)
        intersected_columns = get_intersected_box_index(cell, column_boxes)
        return intersected_rows, intersected_columns

    def _find_first_intersected_element(self, intersected_els: tuple) -> int:
        ind = -1
        elements_compared = [element[1] for element in intersected_els if element[0] > self.AREA_THRESHOLD]
        if elements_compared:
            ind = min(elements_compared)
        return ind

    def _check_replace_cell(self, col: int, row: int, index: List) -> List:
        cell0 = [col, row, None]
        for cell in index:
            if col == cell[0] and row == cell[1] and cell0[2] is None:
                cell0 = cell
            elif col == cell[0] and row == cell[1] and cell0[2] is not None:
                cell0 = self._generate_new_cell(cell, cell0)
        return cell0

    def _merge_multiple_cells(self, index: List) -> Tuple:
        max_col = max([line[0] for line in index]) + 1
        max_row = max([line[1] for line in index]) + 1
        indexed_list = []
        for col in range(max_col):
            for row in range(max_row):
                cellnew = self._check_replace_cell(col, row, index)
                indexed_list.append(cellnew)
        model_cells_list = list(filter(lambda x: x[2] is not None, indexed_list))
        undetected_cells_list = list(filter(lambda x: x[2] is None, indexed_list))
        return model_cells_list, undetected_cells_list

    def _generate_borders(self, index: List) -> List:
        for cell in index:
            columns = self._table_elements.columns[cell[0]]
            rows = self._table_elements.rows[cell[1]]
            cell[2] = (columns[0], rows[0], columns[1], rows[1])
        return index


def cell_tuple_replace(cell_to: CellTuple, cell_from: CellTuple, position_to: int, position_from: int) -> CellTuple:
    cell_to_list = list(cell_to)
    cell_to_list[position_to] = cell_from[position_from]
    return (cell_to_list[0], cell_to_list[1], cell_to_list[2], cell_to_list[3])


def detect_lines_with_cv2(table_elements: TableElements, ocr, is_pdf_searchble):
    cell_threshold = 0.3
    final_cells_list = []
    not_model_cells_list = create_table_from_lines(table_elements.columns, table_elements.rows)
    new_cell_and_content = cell_to_content(
        list(not_model_cells_list.keys()),
        ocr,
        table_elements.table[1],
        table_elements.table[0],
        cell_threshold,
    )
    for key, cell_cont in new_cell_and_content.items():
        text_group_by_line = dd(list)
        for cell_text, box, text_line in zip(cell_cont["cell_content"], cell_cont["cell_boxes"], cell_cont["line"]):
            text_group_by_line[text_line].append([cell_text, box])

        sorted_text = ""
        for line in sorted(text_group_by_line.keys(), reverse=is_pdf_searchble):
            line_cont = text_group_by_line[line]
            line_cont = sorted(line_cont, key=lambda x: x[1][0])
            tmp = " ".join([cont[0] for cont in line_cont])
            sorted_text = f" {tmp}"
        cell_obj = Cell(column=not_model_cells_list[key]["col"], row=not_model_cells_list[key]["row"], value=sorted_text)
        final_cells_list.append(cell_obj)
    return sorted(final_cells_list, key=lambda x: (x.row, x.column))


def building_output(
    table_elements: TableElements,
    ocr,
    lines_detection_method: str,
    is_pdf_searchble: bool,
    image_shape: Tuple[int, int],
) -> TableEntity:

    if lines_detection_method == "cv2":
        table_elements.cells = detect_lines_with_cv2(table_elements, ocr, is_pdf_searchble)
    else:
        # lines_detection_method is not "cv2"
        detector = LineDetector(table_elements, ocr, is_pdf_searchble)
        cells_list, new_columns = detector.detect_lines()
        table_elements.cells = cells_list
        table_elements.columns = new_columns
    return build_relative_table(table_elements, image_shape)


def delete_empty_rows_columns(cellist: List[Cell]) -> List[Cell]:
    # implemented incorrectly
    cols = {cell.column for cell in cellist}
    rows = {cell.row for cell in cellist}
    cols_dict = dict(zip(cols, list(range(len(cols)))))
    rows_dict = dict(zip(rows, list(range(len(rows)))))
    for cell in cellist:
        cell.column = cols_dict[cell.column]
        cell.row = rows_dict[cell.row]
    return cellist


def expand_columns(columns_boxes: List[CellTuple], table_box: Bbox) -> List[Interval]:
    column_starts = list(map(itemgetter(0), columns_boxes))[1:]
    return list(zip([0] + column_starts, column_starts + [table_box[2]]))


def cells_start_list(column_table: np.ndarray, subcolumn_index: int, last=True) -> List:
    cell_index = 1 if last is True else 0
    coordlist = column_table[:, subcolumn_index, cell_index].tolist()
    return [a for a in coordlist if a is not None]


class HeuristicColumnDetector:
    def __init__(self, index: List, old_columns: List[Interval], threshold: float):
        self.index = index
        self.maxcolumns = 0
        self.maxrows = 0
        self.old_columns: List[Interval] = old_columns
        self.threshold = threshold

    def detect_columns(self) -> List[Interval]:
        columns, rows, _ = zip(*self.index)
        self.maxcolumns = max(columns)
        self.maxrows = max(rows)
        table = self._table_index_to_array()
        full_border_list = self._generate_new_borders(table)
        return self._create_column_border_pairs(full_border_list)

    def _table_index_to_array(self) -> np.ndarray:
        table = np.empty([self.maxcolumns + 1, self.maxrows + 1], dtype=object)
        for element in self.index:
            cell_value = (element[2][0], element[2][2])
            if table[element[0]][element[1]] is None:
                table[element[0]][element[1]] = [cell_value]
            else:
                table[element[0]][element[1]].append(cell_value)
        return table

    def _column_to_array(self, column: List, max_cells: int):
        column_table = np.empty([self.maxrows + 1, max_cells, 2], dtype=object)
        for intersection_index, intersection in enumerate(column):
            if intersection is not None:
                for coordinate_index, coordinate in enumerate(sorted(intersection)):
                    column_table[intersection_index][coordinate_index][0] = coordinate[0]
                    column_table[intersection_index][coordinate_index][1] = coordinate[1]
        for ind_row, row in enumerate(column_table):
            for ind_cell, cell in enumerate(row):
                column_table = move_to_next_cell(column_table, ind_row, ind_cell, cell, max_cells)
        return column_table

    def _detect_subcolumn_start(self, column_table: np.ndarray, new_column) -> int:
        column_start = None
        cells_number = len(cells_start_list(column_table, new_column)) / self.maxrows
        if cells_number > self.threshold:
            column_start = min(cells_start_list(column_table, new_column, last=False))
        return column_start

    def _generate_new_borders(self, table: np.ndarray) -> List[List]:
        border_list = []
        for column in table:
            new_borders: List[Any] = []
            max_cells = max([len(cell) for cell in column if cell is not None], default=0)
            if max_cells == 1:
                border_list.append(new_borders)
                continue

            column_table = self._column_to_array(column, max_cells)
            for new_column in range(max_cells):
                column_start = self._detect_subcolumn_start(column_table, new_column)
                if column_start is not None:
                    new_borders.append(column_start)
            new_borders = new_borders[1:]
            border_list.append(new_borders)
        return border_list

    def _create_column_border_pairs(self, full_border_list: List) -> List[Interval]:
        if len(full_border_list) != len(self.old_columns):
            logger.info("column number mismatch")
            return self.old_columns

        new_columns = []
        for ind, borders in enumerate(self.old_columns):
            if full_border_list[ind]:
                border_pairs = zip([borders[0]] + full_border_list[ind], full_border_list[ind] + [borders[1]])
                new_columns.extend(list(border_pairs))
            else:
                new_columns.append(borders)
        return new_columns


def move_to_next_cell(column_table, ind_row, ind_cell, cell, max_cells):
    if ind_cell + 1 < max_cells:
        if cell[0]:
            start_coord_greater = cell[0] > min(cells_start_list(column_table, ind_cell))
        else:
            start_coord_greater = False
        next_cell_is_empty = column_table[ind_row, ind_cell + 1, 0] is None
        if start_coord_greater and next_cell_is_empty:
            column_table[ind_row, ind_cell + 1] = column_table[ind_row, ind_cell]
            column_table[ind_row, ind_cell] = [None, None]
    return column_table
