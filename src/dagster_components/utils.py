from collections.abc import Sequence
from typing import Literal

import pandas as pd

from dagster_components.types import BoundDFType


def cast_all_columns_to_numeric(
    df: BoundDFType,
    ignore: Sequence[str] | None = None,
    *,
    errors: Literal["coerce", "raise"] = "raise",
    make_valid_int: bool = False,
) -> BoundDFType:
    """
    Convert all columns in a DataFrame to numeric types.
    This function attempts to cast all columns in a DataFrame to numeric types,
    with options to ignore specific columns and handle conversion errors.

    Parameters
    ----------
    df : BoundDFType
        The input DataFrame to convert.
    ignore : Sequence[str], optional
        Column names to skip during conversion. Default is None.
    errors : {"coerce", "raise"}, default "raise"
        - "raise": raise an exception if conversion fails
        - "coerce": convert invalid values to NaN
    make_valid_int : bool, default False
        If True, convert columns with all integer values (no decimals)
        to int type. Only applied if all values are non-null and are
        whole numbers.

    Returns
    -------
    BoundDFType
        A new DataFrame with converted columns. The original DataFrame
        is not modified.

    Notes
    -----
    - A copy of the input DataFrame is created before modifications.
    - If make_valid_int is True, columns are converted to int only if
      they contain no missing values and all numeric values are whole numbers.

    Examples
    --------
    >>> df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['1.5', '2.5', '3.5']})
    >>> cast_all_columns_to_numeric(df)
       A    B
    0  1  1.5
    1  2  2.5
    2  3  3.5
    >>> cast_all_columns_to_numeric(df, ignore=['B'], make_valid_int=True)
       A    B
    0  1  1.5
    1  2  2.5
    2  3  3.5
    """

    if ignore is None:
        ignore = []

    df = df.copy()
    for col in df.columns:
        if col not in ignore:
            new_col = pd.to_numeric(df[col], errors=errors)
            if (
                make_valid_int
                and new_col.notna().all()
                and (new_col.to_numpy() % 1 == 0).all()
            ):
                new_col = new_col.astype(int)
            df[col] = new_col

    return df
