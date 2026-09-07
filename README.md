# cfc-dagster-utils

Utilities and IO managers used by CFC Dagster projects.

## Installation

The base package contains the pandas, GeoPandas, and SQLAlchemy utilities and can be
installed without Dagster:

```shell
pip install cfc_dagster_utils
```

Install the extra for the feature you use:

```shell
pip install 'cfc_dagster_utils[dagster]'
pip install 'cfc_dagster_utils[earthengine]'
pip install 'cfc_dagster_utils[postgres]'
pip install 'cfc_dagster_utils[xarray]'
```

The `earthengine` and `xarray` extras include Dagster because their managers depend on
it. Extras can be combined, for example:

```shell
pip install 'cfc_dagster_utils[earthengine,xarray]'
```

## Imports

Optional manager classes are imported from their feature modules. The package and the
`managers` namespace can always be imported without installing any extras.

```python
from cfc_dagster_utils.managers.dataframe import DataFrameFileManager
from cfc_dagster_utils.managers.earthengine import EarthEngineManager
from cfc_dagster_utils.managers.geodataframe import GeoDataFrameFileManager
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.managers.xarray import DataArrayFileManager
```

Importing a feature module without its dependency raises an `ImportError` containing
the exact extra installation command. Import errors originating inside an installed
dependency are preserved instead of being treated as a missing optional dependency.

## Local testing

Run the complete optional-dependency matrix locally with:

```shell
uv run python scripts/test_optional_dependency_matrix.py
```

Each combination runs in a temporary isolated environment, so packages installed in
the project's `.venv` cannot hide a missing optional dependency. GitHub Actions calls
the same runner, keeping the local and CI matrices in sync.

## PostgreSQL and PostGIS outputs

Install the `postgres` extra to use psycopg 3, GeoAlchemy2, the PostgreSQL resource,
and the pandas/GeoPandas IO manager:

```shell
pip install 'cfc_dagster_utils[postgres]'
```

The IO manager is a write-only bridge for placing Python-created frames in PostgreSQL.
Each output declares its destination with flat `schema` and `table` metadata. Any
schema is supported and is created when it does not already exist. Existing tables are
replaced.

```python
import geopandas as gpd

import dagster as dg
from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource


@dg.asset(
    io_manager_key="postgres_io_manager",
    metadata={"schema": "staging", "table": "cities_prepared"},
)
def cities_prepared() -> gpd.GeoDataFrame:
    return load_cities()


postgres_resource = PostgresResource(
    host=dg.EnvVar("POSTGRES_HOST"),
    port=5432,
    database=dg.EnvVar("POSTGRES_DB"),
    user=dg.EnvVar("POSTGRES_USER"),
    password=dg.EnvVar("POSTGRES_PASSWORD"),
)

defs = dg.Definitions(
    assets=[cities_prepared],
    resources={
        "postgres_resource": postgres_resource,
        "postgres_io_manager": PostgresIOManager(
            postgres_resource=postgres_resource,
        ),
    },
)
```

GeoDataFrames use their active geometry column and CRS when written through
`GeoDataFrame.to_postgis()`. The manager does not create constraints or explicitly
manage indexes. Use a database transformation framework such as dbt for durable models,
contracts, constraints, indexes, and SQL transformations.

`PostgresIOManager` does not load tables into downstream Python assets. Use
`PostgresResource.connect()` for an explicit database read when one is needed.
