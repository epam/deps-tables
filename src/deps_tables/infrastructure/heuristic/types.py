from typing import Any, List, Optional, Sequence, Tuple, TypedDict

from numpy.typing import NDArray
from pydantic import BaseModel

OCROutput = TypedDict(
    "OCROutput",
    {
        "bbox": List[int],
        "content": str,
        "line": int,
    },
)

Bbox = Sequence[int]
Interval = Tuple[int, int]
CellTuple = Tuple[int, int, int, int]


class Cell(BaseModel):
    column: int
    row: int
    value: Optional[str] = None


class TableElements(BaseModel):
    rows: List[Interval]
    columns: List[Interval]
    cells: List[Any]
    table: List[int]
    image: NDArray

    class Config:
        arbitrary_types_allowed = True
