from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

G = TypeVar("G", bound="pd.DataFrame")
