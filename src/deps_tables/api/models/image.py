from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from deps_tables.domain.entities import (
    ImageEntity,
    ImageMetadataEntity,
    PointEntity,
    RectangleEntity,
    TextLineEntity,
    WordBoxEntity,
)
from deps_tables.domain.entities.geom import Shape
from deps_tables.extras.ocr import TextLine as TextLineModel


def convert_textline_to_domain(textline: TextLineModel) -> TextLineEntity:
    return TextLineEntity(
        id=textline.id,
        word_boxes=[
            WordBoxEntity(
                content=word_box.content,
                bbox=RectangleEntity(
                    left_top_point=PointEntity(
                        x=word_box.bbox.left_top_point.x,
                        y=word_box.bbox.left_top_point.y,
                    ),
                    right_bottom_point=PointEntity(
                        x=word_box.bbox.right_bottom_point.x,
                        y=word_box.bbox.right_bottom_point.y,
                    ),
                ),
                confidence=word_box.confidence,
            )
            for word_box in textline.word_boxes
        ],
    )


class ImageMetadataModel(BaseModel):
    textlines: Optional[List[TextLineModel]] = None

    def to_domain(self) -> ImageMetadataEntity:
        return ImageMetadataEntity(
            textlines=(
                [convert_textline_to_domain(textline) for textline in self.textlines] if self.textlines is not None else None
            ),
        )


class ImageModel(BaseModel):
    image: np.ndarray
    blob_path: Optional[str] = None
    meta: Optional[ImageMetadataModel] = None

    class Config:
        arbitrary_types_allowed = True

    def to_domain(self) -> ImageEntity:
        return ImageEntity(
            image=self.image,
            blob_path=self.blob_path,
            meta=self.meta.to_domain() if self.meta else None,
        )

    @property
    def shape(self) -> Shape[int]:
        return Shape(
            height=self.image.shape[0],
            width=self.image.shape[1],
        )
