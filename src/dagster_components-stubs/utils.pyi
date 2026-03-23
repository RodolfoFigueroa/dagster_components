from dagster_components.types import G
from typing import Sequence

def cast_all_columns_to_numeric(df: G, ignore: Sequence[str] | None = None) -> G: ...
