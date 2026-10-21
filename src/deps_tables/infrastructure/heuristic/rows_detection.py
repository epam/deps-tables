import warnings
from typing import List

import numpy as np
from sklearn.neighbors import KernelDensity

from .types import CellTuple, Interval

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

Row = int


class RowsIdentifier:

    KERNEL = "gaussian"
    BANDWIDTH = 2.0
    MIN_DATAPOINTS_COUNT = 3
    N_SAMPLES = 500

    def identify_borders(self, img: np.ndarray, cells: List[CellTuple]) -> List[Interval]:
        rows = self._identify_rows(img, cells)
        rows = self._easy_rows_correction(rows, cells)
        if not rows:
            return []
        intervals = []
        for ind, row in enumerate(rows):
            if ind == len(rows) - 1:
                break
            intervals.append((row, rows[ind + 1]))
        return intervals

    def _identify_rows(self, img: np.ndarray, cells: List[CellTuple]) -> List[Row]:
        img_vector, text_boxes_filtered = img, cells
        kde = KernelDensity(kernel=self.KERNEL, bandwidth=self.BANDWIDTH)
        rows: List[Row] = []

        if len(text_boxes_filtered) < self.MIN_DATAPOINTS_COUNT:
            return rows

        h, w = img_vector.shape[:2]
        t = [1, 1, w - 1, h - 1]
        kde.fit(np.array([b[3] for b in text_boxes_filtered]).reshape(-1, 1))
        xs = np.linspace(t[1], t[3], self.N_SAMPLES)
        ys = 2 ** kde.score_samples(xs.reshape(-1, 1))
        maxima = [int(element) for element in self._find_maxima(xs, ys)]
        if maxima:
            rows.extend(maxima)
        rows = [max(0, row) for row in rows]
        # add first row
        rows = [0] + rows
        return sorted(rows)

    def _find_maxima(self, xs: np.ndarray, ys: np.ndarray) -> List[float]:
        if len(xs) != len(ys):
            raise Exception("Comparing arrays of different lengths.")  # noqa: WPS454
        return [
            float(xs[i])
            for i in range(1, len(xs) - 1)
            if ys[i] > 1e-2 and ys[i - 1] <= ys[i] and ys[i + 1] <= ys[i]  # noqa: WPS221, WPS459
        ]

    def _easy_rows_correction(self, detected_rows: List[Row], cells: List[CellTuple]) -> List[Row]:
        # new post-proc
        intervals = []
        for ind in range(len(detected_rows) - 1):
            intervals.append([detected_rows[ind], detected_rows[ind + 1]])

        min_cell_hight = min([cell[3] - cell[1] for cell in cells])
        indices_to_delete = set()
        for ind, interval in enumerate(intervals):  # noqa: WPS440
            if interval[1] - interval[0] < min_cell_hight - 3:
                indices_to_delete.add(ind)
        return [row for idx, row in enumerate(detected_rows) if idx not in indices_to_delete]

    def _rows_correction(self, detected_rows: List[Row], cells: List[CellTuple]) -> List[Row]:
        if not cells:
            return []
        min_y = min([box[1] for box in cells])
        rows = [min_y]
        if detected_rows:
            rows.extend(detected_rows)
        else:
            max_y = max([box[1] for box in cells])
            rows.append(max_y)
        ok_rows = rows.copy()
        corrected = []
        for row in rows:
            current_row = row
            appended_row = 0
            bad_row = False
            while self._intersec_row_box(current_row, cells):
                bad_row = True
                res = max(self._intersec_row_box(current_row, cells))
                current_row = res
                appended_row = res
            if bad_row:
                ok_rows.remove(row)
                corrected.append(appended_row)

        corrected = sorted(set(corrected))
        ok_rows.extend(corrected)

        return ok_rows

    def _intersec_row_box(self, row: Row, cells: List[CellTuple]) -> List[Row]:
        res = []
        for box in cells:
            if box[1] < row < box[3]:
                res.append(box[3])
        return res
