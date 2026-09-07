from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

import dagster as dg
import geopandas as gpd
import pandas as pd
import pytest
import sqlalchemy
from shapely.geometry import Point
from sqlalchemy.dialects import postgresql

from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource

TEST_PASSWORD = "secret"  # noqa: S105 - inert test credential.
SPECIAL_TEST_PASSWORD = "p@ss/w:rd"  # noqa: S105 - URL-encoding fixture.


def _resource() -> PostgresResource:
    return PostgresResource(
        host="db.example",
        port=5432,
        user="user",
        password=TEST_PASSWORD,
        database="warehouse",
    )


def _connection() -> MagicMock:
    conn = MagicMock(spec=sqlalchemy.Connection)
    conn.dialect = postgresql.dialect()
    return conn


def _output_context(
    metadata: dict[str, object] | None = None,
) -> MagicMock:
    context = MagicMock(spec=dg.OutputContext)
    context.definition_metadata = (
        metadata if metadata is not None else {"schema": "analytics", "table": "target"}
    )
    return context


def _manager_with_connection() -> tuple[
    PostgresIOManager,
    PostgresResource,
    MagicMock,
    MagicMock,
]:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)
    conn = _connection()
    transaction = MagicMock(spec=AbstractContextManager)
    transaction.__enter__.return_value = conn
    return manager, resource, conn, transaction


def _statement_strings(conn: MagicMock) -> list[str]:
    return [
        str(call.args[0].compile(dialect=conn.dialect)).strip()
        for call in conn.execute.call_args_list
    ]


def test_resource_builds_psycopg_url_and_disposes_engine() -> None:
    resource = PostgresResource(
        host="db.example",
        port=5432,
        user="user@tenant",
        password=SPECIAL_TEST_PASSWORD,
        database="warehouse",
    )
    engine = MagicMock(spec=sqlalchemy.Engine)

    with patch("sqlalchemy.create_engine", return_value=engine) as create_engine:
        resource.setup_for_execution(MagicMock())

    url = create_engine.call_args.args[0]
    assert isinstance(url, sqlalchemy.URL)
    assert url.drivername == "postgresql+psycopg"
    assert url.username == "user@tenant"
    assert url.password == SPECIAL_TEST_PASSWORD

    resource.teardown_after_execution(MagicMock())
    engine.dispose.assert_called_once_with()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"schema": "analytics"},
        {"table": "target"},
        {"schema": "", "table": "target"},
        {"schema": "   ", "table": "target"},
        {"schema": 1, "table": "target"},
        {"schema": "analytics", "table": ""},
        {"schema": "analytics", "table": "   "},
        {"schema": "analytics", "table": 1},
    ],
)
def test_manager_requires_explicit_non_empty_destination_metadata(
    metadata: dict[str, object],
) -> None:
    manager, _resource_unused, _, _ = _manager_with_connection()

    with (
        patch.object(PostgresResource, "begin") as begin,
        pytest.raises(ValueError, match="metadata"),
    ):
        manager.handle_output(_output_context(metadata), pd.DataFrame({"id": [1]}))

    begin.assert_not_called()


def test_manager_replaces_dataframe_in_arbitrary_schema() -> None:
    manager, _resource_unused, conn, transaction = _manager_with_connection()
    context = _output_context({"schema": "custom schema", "table": "target table"})
    frame = pd.DataFrame({"id": [1, 2]})

    with (
        patch.object(PostgresResource, "begin", return_value=transaction),
        patch.object(pd.DataFrame, "to_sql") as to_sql,
    ):
        manager.handle_output(context, frame)

    assert _statement_strings(conn) == ['CREATE SCHEMA IF NOT EXISTS "custom schema"']
    to_sql.assert_called_once_with(
        "target table",
        conn,
        schema="custom schema",
        if_exists="replace",
        index=False,
    )
    context.add_output_metadata.assert_called_once_with(
        {
            "schema": "custom schema",
            "table": "target table",
            "relation": "custom schema.target table",
            "row_count": 2,
        }
    )


def test_manager_uses_geodataframe_writer_and_infers_geometry_column() -> None:
    manager, _resource_unused, conn, transaction = _manager_with_connection()
    context = _output_context()
    frame = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    frame.rename_geometry("geom", inplace=True)

    with (
        patch.object(PostgresResource, "begin", return_value=transaction),
        patch.object(gpd.GeoDataFrame, "to_postgis") as to_postgis,
        patch.object(pd.DataFrame, "to_sql") as to_sql,
    ):
        manager.handle_output(context, frame)

    to_postgis.assert_called_once_with(
        "target",
        conn,
        schema="analytics",
        if_exists="replace",
        index=False,
    )
    to_sql.assert_not_called()
    context.add_output_metadata.assert_called_once_with(
        {
            "schema": "analytics",
            "table": "target",
            "relation": "analytics.target",
            "row_count": 1,
            "geometry_column": "geom",
        }
    )


def test_manager_rejects_non_string_geometry_column_before_connecting() -> None:
    manager, _resource_unused, _, _ = _manager_with_connection()
    frame = gpd.GeoDataFrame(
        {0: [Point(0, 0)]},
        geometry=0,
        crs="EPSG:4326",
    )

    with (
        patch.object(PostgresResource, "begin") as begin,
        pytest.raises(TypeError, match="geometry column names must be strings"),
    ):
        manager.handle_output(_output_context(), frame)

    begin.assert_not_called()


def test_manager_rejects_unknown_output_type_before_connecting() -> None:
    manager, _resource_unused, _, _ = _manager_with_connection()

    with (
        patch.object(PostgresResource, "begin") as begin,
        pytest.raises(TypeError, match="Unsupported PostgreSQL output type: dict"),
    ):
        manager.handle_output(_output_context(), {})  # ty: ignore[invalid-argument-type]

    begin.assert_not_called()


def test_manager_propagates_write_failure_through_transaction() -> None:
    manager, _resource_unused, _, transaction = _manager_with_connection()
    frame = pd.DataFrame({"id": [1]})

    with (
        patch.object(PostgresResource, "begin", return_value=transaction),
        patch.object(
            pd.DataFrame,
            "to_sql",
            side_effect=RuntimeError("write failed"),
        ),
        pytest.raises(RuntimeError, match="write failed"),
    ):
        manager.handle_output(_output_context(), frame)

    assert transaction.__exit__.call_args.args[0] is RuntimeError


def test_manager_rejects_input_loading_without_connecting() -> None:
    resource = _resource()
    manager = PostgresIOManager(postgres_resource=resource)

    with (
        patch.object(PostgresResource, "connect") as connect,
        pytest.raises(NotImplementedError, match="output-only"),
    ):
        manager.load_input(MagicMock(spec=dg.InputContext))

    connect.assert_not_called()
