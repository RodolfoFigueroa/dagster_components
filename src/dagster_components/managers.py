from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, Literal, overload

import dagster as dg
import geopandas as gpd
import pandas as pd
import sqlalchemy
from dagster._config.pythonic_config.resource import TResValue

from dagster_components.resources import PostgresResource
from dagster_components.types import DFType, T


class _DataFrameBasePostgresManager(
    dg.ConfigurableIOManager,
    Generic[DFType, TResValue],
):
    """Base class for Dagster IO managers that read and write DataFrames to/from PostgreSQL.

    Subclasses must implement ``write_table`` and ``load_table`` to define how data is
    serialized and deserialized for a specific DataFrame type.

    Attributes:
        postgres_resource: A Dagster resource dependency providing a PostgreSQL connection.
    """

    postgres_resource: dg.ResourceDependency[PostgresResource]

    def write_table(
        self,
        df: DFType,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        """Write a DataFrame to a PostgreSQL table.

        Must be implemented by subclasses.

        Args:
            df: The DataFrame to write.
            table_name: The name of the destination table.
            conn: An active SQLAlchemy database connection.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        msg = "write_table must be implemented by subclasses"
        raise NotImplementedError(msg)

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> DFType:
        """Load data from a PostgreSQL table into a DataFrame.

        Must be implemented by subclasses.

        Args:
            table_name: The name of the source table.
            cols_str: A comma-separated string of column names to select, or ``"*"`` for all columns.
            conn: An active SQLAlchemy database connection.

        Returns:
            The loaded DataFrame.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        msg = "load_table must be implemented by subclasses"
        raise NotImplementedError(msg)

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: DFType,
    ) -> None:
        """Persist a DataFrame as a PostgreSQL table.

        Writes the DataFrame to the table specified in
        ``context.definition_metadata["table_name"]``. Optionally adds a primary key
        constraint and foreign key constraints based on additional metadata keys.

        Args:
            context: The Dagster output context. Relevant metadata keys:
                - ``table_name`` (str, required): Target table name.
                - ``primary_key`` (str, optional): Column to set as the primary key.
                - ``foreign_keys`` (list[dict], optional): List of foreign key mappings,
                  each with keys ``column``, ``ref_table``, and ``ref_column``.
            obj: The DataFrame to write.

        Raises:
            ValueError: If the specified primary key or a foreign key column is not
                present in the DataFrame's columns.
        """
        table = context.definition_metadata["table_name"]

        with self.postgres_resource.connect() as conn:
            self.write_table(obj, table, conn)

            if "primary_key" in context.definition_metadata:
                primary_key = context.definition_metadata["primary_key"]

                if primary_key not in obj.columns:
                    err = f"Primary key {primary_key} not found in DataFrame columns"
                    raise ValueError(err)

                conn.execute(
                    sqlalchemy.text(
                        f'ALTER TABLE {table} ADD PRIMARY KEY ("{primary_key}");',
                    ),
                )

            if "foreign_keys" in context.definition_metadata:
                foreign_keys = context.definition_metadata["foreign_keys"]

                for fk_map in foreign_keys:
                    fk_col = fk_map["column"]
                    ref_table = fk_map["ref_table"]
                    ref_col = fk_map["ref_column"]

                    if fk_col not in obj.columns:
                        err = f"Foreign key column {fk_col} not found in DataFrame columns."
                        raise ValueError(err)

                    conn.execute(
                        sqlalchemy.text(
                            f"""
                            ALTER TABLE {table}
                            ADD FOREIGN KEY ("{fk_col}")
                            REFERENCES {ref_table}("{ref_col}")
                            """,
                        ),
                    )

            conn.commit()

    def load_input(self, context: dg.InputContext) -> DFType:
        """Load a DataFrame from the PostgreSQL table written by the upstream output.

        Reads from the table specified in the upstream output's ``table_name`` metadata.
        If the input metadata contains a ``columns`` key, only those columns are selected.

        Args:
            context: The Dagster input context. Relevant metadata keys:
                - ``columns`` (list[str], optional): Specific columns to load. Defaults to all columns.

        Returns:
            The loaded DataFrame.

        Raises:
            ValueError: If no upstream output is found.
        """
        upstream_output = context.upstream_output
        if upstream_output is None:
            err = "No upstream output found."
            raise ValueError(err)

        table = upstream_output.definition_metadata["table_name"]

        in_metadata = context.definition_metadata
        if "columns" in in_metadata:
            wanted_cols = in_metadata["columns"]
            cols_str = ", ".join(wanted_cols)
        else:
            cols_str = "*"

        with self.postgres_resource.connect() as conn:
            return self.load_table(table, cols_str, conn)


class DataFramePostgresManager(_DataFrameBasePostgresManager[pd.DataFrame, Any]):
    """Dagster IO manager for reading and writing pandas DataFrames to/from PostgreSQL.

    Uses pandas ``to_sql`` and ``read_sql`` for serialization and deserialization.
    Inherits output and input handling (including primary and foreign key constraints)
    from ``_DataFrameBasePostgresManager``.
    """

    def write_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        """Write a pandas DataFrame to a PostgreSQL table using ``to_sql``.

        Replaces the table if it already exists.

        Args:
            df: The pandas DataFrame to write.
            table_name: The name of the destination table.
            conn: An active SQLAlchemy database connection.
        """
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> pd.DataFrame:
        """Load a pandas DataFrame from a PostgreSQL table using ``read_sql``.

        Args:
            table_name: The name of the source table.
            cols_str: A comma-separated string of column names to select, or ``"*"`` for all columns.
            conn: An active SQLAlchemy database connection.

        Returns:
            The loaded DataFrame.
        """
        return pd.read_sql(f"SELECT {cols_str} FROM {table_name}", conn)  # noqa: S608


class GeoDataFramePostGISManager(_DataFrameBasePostgresManager[gpd.GeoDataFrame, Any]):
    """Dagster IO manager for reading and writing GeoDataFrames to/from PostGIS.

    Uses geopandas ``to_postgis`` and ``read_postgis`` for serialization and
    deserialization. Assumes a ``geometry`` column is present for spatial data.
    Inherits output and input handling from ``_DataFrameBasePostgresManager``.
    """

    def write_table(
        self,
        df: gpd.GeoDataFrame,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        """Write a GeoDataFrame to a PostGIS table using ``to_postgis``.

        Replaces the table if it already exists.

        Args:
            df: The GeoDataFrame to write.
            table_name: The name of the destination table.
            conn: An active SQLAlchemy database connection.
        """
        df.to_postgis(table_name, conn, if_exists="replace")

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> gpd.GeoDataFrame:
        """Load a GeoDataFrame from a PostGIS table using ``read_postgis``.

        Assumes the geometry column is named ``geometry``.

        Args:
            table_name: The name of the source table.
            cols_str: A comma-separated string of column names to select, or ``"*"`` for all columns.
            conn: An active SQLAlchemy database connection.

        Returns:
            The loaded GeoDataFrame.
        """
        return gpd.read_postgis(
            f"SELECT {cols_str} FROM {table_name}",  # noqa: S608
            conn,
            geom_col="geometry",
        )


class PathResource(dg.ConfigurableResource):
    """Dagster resource providing a base output directory path for file-based IO managers.

    Attributes:
        out_path: The root directory path where assets are stored.
    """

    out_path: str


class _DataFrameBaseFileManager(dg.ConfigurableIOManager):
    """Base class for Dagster IO managers that read and write DataFrames to/from files.

    Handles path resolution for both partitioned and non-partitioned assets.
    Subclasses must implement ``handle_output`` and ``load_input``.

    Attributes:
        path_resource: A resource dependency providing the root output directory path.
        extension: The file extension to use. One of ``.parquet``, ``.csv``, ``.gpkg``,
            or ``.geoparquet``.
    """

    path_resource: dg.ResourceDependency[PathResource]
    extension: Literal[".parquet", ".csv", ".gpkg", ".geoparquet"]

    def _get_single_partition_key_path(
        self, partition_key: str, asset_dir: Path
    ) -> Path:
        """Resolve the file path for a single partition key within an asset directory.

        Partition keys containing ``|`` are treated as multi-dimensional and mapped to
        nested subdirectory segments.

        Args:
            partition_key: The partition key string, with ``|`` as segment separator.
            asset_dir: The root directory for the asset.

        Returns:
            The resolved file path including the configured extension.

        Raises:
            ValueError: If ``asset_dir`` does not exist or is not a directory.
        """
        if not asset_dir.is_dir():
            err = f"Asset directory {asset_dir} does not exist or is not a directory."
            raise ValueError(err)

        segments = partition_key.split("|")
        fpath = asset_dir / "/".join(segments)
        return fpath.with_suffix(fpath.suffix + self.extension)

    @overload
    def _get_partitioned_asset_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        asset_dir: Path,
        *,
        allow_multiple_partitions: Literal[False],
    ) -> Path: ...

    @overload
    def _get_partitioned_asset_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        asset_dir: Path,
        *,
        allow_multiple_partitions: Literal[True],
    ) -> dict[str, Path]: ...

    def _get_partitioned_asset_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        asset_dir: Path,
        *,
        allow_multiple_partitions: bool,
    ) -> Path | dict[str, Path]:
        """Resolve the file path(s) for a partitioned asset context.

        If the context has a single partition key, returns a single ``Path``.
        If multiple partition keys are present and ``allow_multiple_partitions`` is
        ``True``, returns a mapping of partition key to ``Path``.

        Args:
            context: The Dagster input or output context providing partition key information.
            asset_dir: The root directory for the asset.
            allow_multiple_partitions: Whether multiple partition paths may be returned.

        Returns:
            A single path for single-partition contexts, or a dict mapping partition key
            to path when multiple partitions are present.

        Raises:
            ValueError: If multiple partition keys are present and
                ``allow_multiple_partitions`` is ``False``.
        """
        if len(context.asset_partition_keys) == 1:
            return self._get_single_partition_key_path(
                context.asset_partition_key, asset_dir
            )

        if not allow_multiple_partitions:
            err = (
                "Multiple partition keys found for asset, but allow_multiple_partitions is False. "
                f"Partition keys: {context.asset_partition_keys}"
            )
            raise ValueError(err)

        final_path = {}
        for partition_key in context.asset_partition_keys:
            final_path[partition_key] = self._get_single_partition_key_path(
                partition_key, asset_dir
            )
        return final_path

    @overload
    def _get_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        *,
        allow_multiple_partitions: Literal[False] = False,
    ) -> Path: ...

    @overload
    def _get_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        *,
        allow_multiple_partitions: Literal[True],
    ) -> dict[str, Path]: ...

    def _get_path(
        self,
        context: dg.InputContext | dg.OutputContext,
        *,
        allow_multiple_partitions: bool = False,
    ) -> Path | dict[str, Path]:
        """Resolve the output file path(s) for a given IO context.

        Combines the root path from ``path_resource`` with the asset key path to
        form a directory, then delegates to partition-aware helpers as needed.

        Args:
            context: The Dagster input or output context.
            allow_multiple_partitions: When ``True``, allows returning a
                ``dict[str, Path]`` for assets with multiple active partition keys.
                Defaults to ``False``.

        Returns:
            A single path for non-partitioned or single-partition assets, or a dict
            mapping partition key to path when multiple partitions are present.
        """
        out_path = Path(self.path_resource.out_path)
        asset_dir = out_path / "/".join(context.asset_key.path)

        if context.has_asset_partitions:
            return self._get_partitioned_asset_path(
                context, asset_dir, allow_multiple_partitions=allow_multiple_partitions
            )

        return asset_dir.with_suffix(asset_dir.suffix + self.extension)

    @overload
    def _dispatch_multiple_partitions(
        self,
        fpath: Path,
        func: Callable[[Path], T],
    ) -> T: ...

    @overload
    def _dispatch_multiple_partitions(
        self,
        fpath: dict[str, Path],
        func: Callable[[Path], T],
    ) -> dict[str, T]: ...

    def _dispatch_multiple_partitions(
        self,
        fpath: Path | dict[str, Path],
        func: Callable[[Path], T],
    ) -> T | dict[str, T]:
        """Apply a function to a single path or to each path in a partition mapping.

        Args:
            fpath: Either a single ``Path`` or a dict mapping partition keys to paths.
            func: A callable that accepts a ``Path`` and returns a value of type ``T``.

        Returns:
            The result of ``func(fpath)`` when ``fpath`` is a single ``Path``, or a dict
            mapping partition key to function result when ``fpath`` is a dict.

        Raises:
            TypeError: If ``fpath`` is neither a ``Path`` nor a ``dict``.
        """
        if isinstance(fpath, Path):
            return func(fpath)

        if isinstance(fpath, dict):
            return {k: func(v) for k, v in fpath.items()}

        err = f"Unexpected type for file path: {type(fpath)}"
        raise TypeError(err)

    def handle_output(self, context: dg.OutputContext, obj: DFType) -> None:
        """Write a DataFrame to a file. Must be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        raise NotImplementedError

    def load_input(self, context: dg.InputContext) -> DFType:
        """Load a DataFrame from a file. Must be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        raise NotImplementedError


class DataFrameFileManager(_DataFrameBaseFileManager):
    """Dagster IO manager for reading and writing pandas DataFrames to/from local files.

    Supports ``.parquet`` and ``.csv`` file formats. For partitioned assets, loading
    returns a mapping of partition key to DataFrame.
    """

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        """Write a pandas DataFrame to a file.

        Creates parent directories as needed. The format is determined by ``extension``.

        Args:
            context: The Dagster output context used to resolve the output file path.
            obj: The pandas DataFrame to write.

        Raises:
            ValueError: If ``extension`` is not ``.parquet`` or ``.csv``.
        """
        fpath = self._get_path(context)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if self.extension == ".parquet":
            obj.to_parquet(fpath)
        elif self.extension == ".csv":
            obj.to_csv(fpath, index=True)
        else:
            err = f"Unsupported file extension: {self.extension}"
            raise ValueError(err)

    def load_input(
        self, context: dg.InputContext
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Load a pandas DataFrame (or a mapping of DataFrames) from a file.

        For non-partitioned or single-partition assets, returns a single DataFrame.
        For multi-partition inputs, returns a dict mapping partition key to DataFrame.

        Args:
            context: The Dagster input context used to resolve the file path(s).

        Returns:
            A single DataFrame or a dict mapping partition key to DataFrame.

        Raises:
            ValueError: If ``extension`` is not ``.parquet`` or ``.csv``.
        """
        fpath = self._get_path(context, allow_multiple_partitions=True)

        if self.extension == ".parquet":
            return self._dispatch_multiple_partitions(fpath, pd.read_parquet)

        if self.extension == ".csv":
            return self._dispatch_multiple_partitions(
                fpath, lambda p: pd.read_csv(p, index_col=0)
            )

        err = f"Unsupported file extension: {self.extension}"
        raise ValueError(err)


class GeoDataFrameFileManager(_DataFrameBaseFileManager):
    """Dagster IO manager for reading and writing GeoDataFrames to/from local files.

    Supports ``.gpkg`` (GeoPackage) and ``.geoparquet`` file formats. For partitioned
    assets, loading returns a mapping of partition key to GeoDataFrame.
    """

    def handle_output(self, context: dg.OutputContext, obj: gpd.GeoDataFrame) -> None:
        """Write a GeoDataFrame to a file.

        Creates parent directories as needed. The format is determined by ``extension``.

        Args:
            context: The Dagster output context used to resolve the output file path.
            obj: The GeoDataFrame to write.

        Raises:
            ValueError: If ``extension`` is not ``.gpkg`` or ``.geoparquet``.
        """
        fpath = self._get_path(context)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if self.extension == ".gpkg":
            obj.to_file(fpath, driver="GPKG")
        elif self.extension == ".geoparquet":
            obj.to_parquet(fpath, index=True)
        else:
            err = f"Unsupported file extension: {self.extension}"
            raise ValueError(err)

    def load_input(
        self, context: dg.InputContext
    ) -> gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame]:
        """Load a GeoDataFrame (or a mapping of GeoDataFrames) from a file.

        For non-partitioned or single-partition assets, returns a single GeoDataFrame.
        For multi-partition inputs, returns a dict mapping partition key to GeoDataFrame.

        Args:
            context: The Dagster input context used to resolve the file path(s).

        Returns:
            A single GeoDataFrame or a dict mapping partition key to GeoDataFrame.

        Raises:
            ValueError: If ``extension`` is not ``.gpkg`` or ``.geoparquet``.
        """
        fpath = self._get_path(context, allow_multiple_partitions=True)

        if self.extension == ".gpkg":
            return self._dispatch_multiple_partitions(fpath, gpd.read_file)

        if self.extension == ".geoparquet":
            return self._dispatch_multiple_partitions(fpath, gpd.read_parquet)

        err = f"Unsupported file extension: {self.extension}"
        raise ValueError(err)
