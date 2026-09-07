from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import pandas as pd
import sqlalchemy

from cfc_dagster_utils._optional import raise_optional_dependency_error
from cfc_dagster_utils.resources import PostgresResource

try:
    import dagster as dg
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="postgres",
    )

try:
    import geoalchemy2  # noqa: F401
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="geoalchemy2",
        dependency_name="geoalchemy2",
        extra="postgres",
    )


class PostgresIOManager(dg.ConfigurableIOManager):
    """Write pandas and GeoPandas outputs to PostgreSQL tables."""

    postgres_resource: dg.ResourceDependency[PostgresResource]

    @staticmethod
    def _destination(metadata: Mapping[str, Any]) -> tuple[str, str]:
        schema = metadata.get("schema")
        table = metadata.get("table")
        if not isinstance(schema, str) or not schema.strip():
            msg = "PostgreSQL outputs require non-empty string metadata 'schema'"
            raise ValueError(msg)
        if not isinstance(table, str) or not table.strip():
            msg = "PostgreSQL outputs require non-empty string metadata 'table'"
            raise ValueError(msg)
        return schema, table

    @staticmethod
    def _ensure_schema(conn: sqlalchemy.Connection, schema: str) -> None:
        quoted_schema = conn.dialect.identifier_preparer.quote_schema(schema)
        conn.execute(sqlalchemy.text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: pd.DataFrame,
    ) -> None:
        """Replace the configured PostgreSQL table with a frame."""
        if not isinstance(obj, pd.DataFrame):
            msg = f"Unsupported PostgreSQL output type: {type(obj).__name__}"
            raise TypeError(msg)

        schema, table = self._destination(context.definition_metadata)
        geometry_column: str | None = None
        if isinstance(obj, gpd.GeoDataFrame):
            geometry_name = obj.geometry.name
            if not isinstance(geometry_name, str):
                msg = "GeoDataFrame geometry column names must be strings"
                raise TypeError(msg)
            geometry_column = geometry_name

        with self.postgres_resource.begin() as conn:
            self._ensure_schema(conn, schema)
            if isinstance(obj, gpd.GeoDataFrame):
                obj.to_postgis(
                    table,
                    conn,
                    schema=schema,
                    if_exists="replace",
                    index=False,
                )
            else:
                obj.to_sql(
                    table,
                    conn,
                    schema=schema,
                    if_exists="replace",
                    index=False,
                )

        metadata: dict[str, str | int] = {
            "schema": schema,
            "table": table,
            "relation": f"{schema}.{table}",
            "row_count": len(obj),
        }
        if geometry_column is not None:
            metadata["geometry_column"] = geometry_column
        context.add_output_metadata(metadata)

    def load_input(self, context: dg.InputContext) -> None:
        """Reject input loading because this manager is output-only."""
        msg = "PostgresIOManager is output-only and cannot load inputs"
        raise NotImplementedError(msg)
