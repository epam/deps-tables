import logging
from itertools import chain, combinations
from typing import List, Tuple

from deps_tables.infrastructure.heuristic import utils as util

logger = logging.getLogger(__name__)

Column = Tuple[int, int, int, int]
Columns = List[Column]


class ColumnsCorrection:
    MERGE_COLUMNS_THRESHOLD = 0.7
    MERGE_CELLS_THRESHOLD = 0.4

    def correct_columns(
        self,
        columns: Columns,
    ) -> Columns:
        merged_columns = util.merge_overlapping_boxes(columns, self.MERGE_COLUMNS_THRESHOLD)
        merged_columns = merge_vertical_neighbours(merged_columns)
        if len(merged_columns) < 2:
            return merged_columns
        sorted_lines = self.sort_columns(merged_columns)
        res_columns = [self.contract_column(column_line) for column_line in sorted_lines]
        return [col for col in chain.from_iterable(res_columns) if util.is_box_valid(col)]

    def sort_columns(self, columns: Columns) -> List[Columns]:
        lines = self.form_lines_from_columns(columns)
        sorted_lines = sorted(lines, key=lambda x: x[0][1])

        def srt(line):
            return sorted(line, key=lambda y: y[0])

        return list(map(srt, sorted_lines))

    def form_lines_from_columns(self, columns: Columns) -> List[Columns]:
        lines: List[Columns] = []
        for bbox1, bbox2 in combinations(columns, 2):
            if util.row_neighbor(bbox1, bbox2):
                if not lines:
                    lines.append([bbox1, bbox2])
                    continue

                for ind, line in enumerate(lines):
                    if util.row_neighbor(bbox1, line[0]):
                        if bbox1 not in line:
                            lines[ind].append(bbox1)
                        if bbox2 not in line:
                            lines[ind].append(bbox2)
                        break
                else:
                    lines.append([bbox1, bbox2])
        return lines

    def contract_column(self, columns: List[Column]) -> List[Column]:
        contracted_columns = []
        if len(columns) < 2:
            return columns
        buf: List[int] = []
        for col_ind, column in enumerate(columns):
            if col_ind == len(columns) - 1:
                if buf:
                    contracted_columns.append((buf[0], column[1], column[2], column[3]))
                else:
                    contracted_columns.append(column)
                break
            left_border = column[2]
            right_border = columns[col_ind + 1][0]
            if left_border > right_border:
                while left_border >= right_border:
                    left_border = left_border - 1
                    right_border = right_border + 1

                if buf:
                    contracted_columns.append((buf[0], column[1], left_border, column[3]))
                    buf.clear()
                else:
                    contracted_columns.append((column[0], column[1], left_border, column[3]))
                buf.append(right_border)

            else:
                if buf:
                    contracted_columns.append((buf[0], column[1], left_border, column[3]))
                    buf.clear()
                else:
                    contracted_columns.append(column)
        return contracted_columns


def merge_vertical_neighbours(columns: Columns) -> Columns:
    if len(columns) < 2:
        return columns
    fullrun = False
    pairfound = True
    while not fullrun and pairfound:
        for bbox1, bbox2 in combinations(columns, 2):
            if util.column_neighbor(bbox1, bbox2):
                columns.append(util.unite_cells(bbox1, bbox2))
                columns.remove(bbox1)
                columns.remove(bbox2)
                pairfound = True
                fullrun = False
                break
            else:
                pairfound = False
            fullrun = True
    return columns
