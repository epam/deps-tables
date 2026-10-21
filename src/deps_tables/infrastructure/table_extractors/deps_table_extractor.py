import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from deps_tables.domain.entities import BoundingBoxEntity, ImageEntity, TableEntity
from deps_tables.domain.interfaces import IBBoxIdentifier, ITableExtractor
from deps_tables.infrastructure.heuristic import utils as util
from deps_tables.infrastructure.heuristic.columns_detection import ColumnsCorrection
from deps_tables.infrastructure.heuristic.rows_detection import RowsIdentifier
from deps_tables.infrastructure.heuristic.types import TableElements
from deps_tables.infrastructure.heuristic.utils import building_output
from deps_tables.infrastructure.services import OCRService

_logger = logging.getLogger(__name__)


class DepsTablesExtractor(ITableExtractor):
    CONFIDENCE_THRESHOLD = 0.85
    MERGE_TABLE_THRESHOLD = 0.4

    def __init__(
        self,
        table_identifier: IBBoxIdentifier,
        column_identifier: IBBoxIdentifier,
        cell_identifier: IBBoxIdentifier,
        ocr_service: OCRService,
    ):
        self._table_identifier = table_identifier
        self._column_identifier = column_identifier
        self._cell_identifier = cell_identifier
        self._ocr_service = ocr_service
        self._row_identifier = RowsIdentifier()
        self._col_correction = ColumnsCorrection()

    def extract_tables(
        self,
        image_obj: ImageEntity,
        ocr_engine: str,
        language: str,
    ) -> List[TableEntity]:
        _logger.info("Starting table extraction with DepsTablesExtractor")
        start = time.time()
        extracted_ocr = self._ocr_service.extract_data(image_obj, ocr_engine, language)
        reformatted_ocr = util.reformat_ocr_output(extracted_ocr)
        ocr = util.resize_ocr_cells(reformatted_ocr)
        is_pdf_searchable = bool(image_obj.meta is not None and image_obj.meta.textlines)
        result = self._extract_structure(image_obj.blob_path, image_obj.image, ocr, is_pdf_searchable)
        if image_obj.meta and image_obj.meta.source_id:
            for table in result:
                table.fill_source_bbox_coordinates(image_obj.meta.source_id)
        _logger.info("Table extraction has completed %s seconds", round(time.time() - start, 3))
        return result

    def _extract_structure(self, img_name: str, image: np.ndarray, ocr: List, is_pdf_searchable: bool) -> List[TableEntity]:
        m_result_table: List[BoundingBoxEntity] = self._table_identifier.identify_borders(image)
        if not m_result_table:
            return []

        result = []
        m_result_table = sorted(
            m_result_table,
            key=lambda bbox: (bbox.left_top_point.y + bbox.right_bottom_point.y, bbox.left_top_point.x),
        )
        data = self._model_output(img_name, image, m_result_table)
        for name in data.keys():
            lines_detection_method = "not cv2"

            if not data[name]["columns"] or not data[name]["cells"]:
                continue

            rows_intervals = self._row_identifier.identify_borders(data[name]["image"], data[name]["cells"])
            if not rows_intervals:
                continue
            sorted_columns = self._col_correction.correct_columns(
                data[name]["columns"],
            )
            column_intervals = util.expand_columns(sorted_columns, data[name]["table"])

            table_elements = TableElements(
                rows=rows_intervals,
                columns=column_intervals,
                cells=data[name]["cells"],
                table=data[name]["table"],
                image=data[name]["image"],
            )
            table_structure = building_output(
                table_elements,
                ocr,
                lines_detection_method,
                is_pdf_searchable,
                image.shape,
            )
            result.append(table_structure)
        return result

    def _model_output(self, name: str, image: np.ndarray, tables: List[BoundingBoxEntity]) -> Dict[str, Any]:
        data = {}
        cut_images = self._process_table_output(name, image, tables)
        merged_boxes = util.merge_overlapping_boxes(
            [cut_images[table]["table_box"] for table in cut_images.keys()],
            self.MERGE_TABLE_THRESHOLD,
        )
        merged_tables = [BoundingBoxEntity.from_tuple((box, self.CONFIDENCE_THRESHOLD + 0.1)) for box in merged_boxes]
        if len(merged_tables) < len(tables):
            cut_images = self._process_table_output(name, image, merged_tables)

        for key, img in cut_images.items():
            columns = self._column_identifier.identify_borders(img["cropped"])
            columns = [col for col in columns if col.confidence >= self.CONFIDENCE_THRESHOLD]
            identified_cells = self._cell_identifier.identify_borders(img["cropped"])
            cells = util.bounding_box_to_tuple(identified_cells)
            cut_cells = util.resize_cells(cells)

            def sort_cell(cell):
                return cell[1] + cell[3], cell[0]

            cut_cells = sorted(cut_cells, key=sort_cell)

            data[key] = {
                "image": img["cropped"],
                "table": img["table_box"],
                "columns": util.bounding_box_to_tuple(columns),
                "cells": cut_cells,
            }
        return data

    def _process_table_output(
        self,
        file_name: str,
        image: np.ndarray,
        m_result_table: List[BoundingBoxEntity],
    ) -> Dict[str, Any]:
        cut_images = {}
        for ind, elem in enumerate(m_result_table):
            if elem.confidence >= self.CONFIDENCE_THRESHOLD:
                left = elem.left_top_point.x
                top = elem.left_top_point.y
                right = elem.right_bottom_point.x
                bottom = elem.right_bottom_point.y

                crop_img = image[top:bottom, left:right]
                cut_images[f"{file_name}_{ind}"] = {
                    "cropped": crop_img,
                    "table_box": [left, top, right, bottom],
                }

        return cut_images

    def _extracting_all_data(self, image: np.ndarray) -> Tuple[List[BoundingBoxEntity], List[BoundingBoxEntity]]:
        m_result_col = self._column_identifier.identify_borders(image)
        m_result_cell = self._cell_identifier.identify_borders(image)
        return m_result_col, m_result_cell
