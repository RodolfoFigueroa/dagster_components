from typing import Any, Generic

import dagster as dg
import geopandas as gpd
import pandas as pd
import sqlalchemy
from dagster._config.pythonic_config.resource import TResValue
from pydantic import PrivateAttr

from dagster_components.types import DFType


class _DataFrameBasePostgresManager(
    dg.ConfigurableIOManager,
    Generic[DFType, TResValue],
):
    host: str
    port: str
    user: str
    password: str
    db: str

    _engine: sqlalchemy.engine.Engine = PrivateAttr()

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:  # noqa: ARG002
        self._engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}?client_encoding=utf8",
        )

    def write_table(
        self,
        df: DFType,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        msg = "write_table must be implemented by subclasses"
        raise NotImplementedError(msg)

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> DFType:
        msg = "load_table must be implemented by subclasses"
        raise NotImplementedError(msg)

    def handle_output(
        self,
        context: dg.OutputContext,
        obj: DFType,
    ) -> None:
        table = context.definition_metadata["table_name"]

        with self._engine.connect() as conn:
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

        with self._engine.connect() as conn:
            return self.load_table(table, cols_str, conn)


class DataFramePostgresManager(_DataFrameBasePostgresManager[pd.DataFrame, Any]):
    def write_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> pd.DataFrame:
        return pd.read_sql(f"SELECT {cols_str} FROM {table_name}", conn)  # noqa: S608


class GeoDataFramePostGISManager(_DataFrameBasePostgresManager[gpd.GeoDataFrame, Any]):
    def write_table(
        self,
        df: gpd.GeoDataFrame,
        table_name: str,
        conn: sqlalchemy.Connection,
    ) -> None:
        df.to_postgis(table_name, conn, if_exists="replace")

    def load_table(
        self,
        table_name: str,
        cols_str: str,
        conn: sqlalchemy.Connection,
    ) -> gpd.GeoDataFrame:
        return gpd.read_postgis(
            f"SELECT {cols_str} FROM {table_name}",  # noqa: S608
            conn,
            geom_col="geometry",
        )
