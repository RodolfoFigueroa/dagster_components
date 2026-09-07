from unittest.mock import MagicMock

import dagster as dg
import pytest

from cfc_dagster_utils.components.postgres import PostgresConnectionComponent
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource

TEST_PASSWORD = "secret"  # noqa: S105 - inert test credential.
TEST_PORT = 5432


def _connection(**kwargs: str) -> PostgresConnectionComponent:
    return PostgresConnectionComponent(
        host="db.example",
        port=TEST_PORT,
        user="user",
        password=TEST_PASSWORD,
        database="warehouse",
        resource_key=kwargs.get("resource_key", "warehouse_resource"),
        io_manager_key=kwargs.get("io_manager_key", "warehouse_manager"),
    )


def test_connection_component_wires_resource_and_io_manager() -> None:
    component = _connection()

    definitions = component.build_defs(MagicMock(spec=dg.ComponentLoadContext))

    resources = definitions.resources
    assert resources is not None
    assert set(resources) == {"warehouse_resource", "warehouse_manager"}
    assert isinstance(resources["warehouse_resource"], PostgresResource)
    assert isinstance(resources["warehouse_manager"], PostgresIOManager)


def test_connection_component_creates_configured_resource() -> None:
    resource = _connection().create_resource()

    assert resource.host == "db.example"
    assert resource.port == TEST_PORT
    assert resource.user == "user"
    assert resource.password == TEST_PASSWORD
    assert resource.database == "warehouse"


def test_connection_component_requires_distinct_resource_keys() -> None:
    with pytest.raises(ValueError, match="must be different"):
        _connection(resource_key="postgres", io_manager_key="postgres")
