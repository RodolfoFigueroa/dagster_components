from dagster_components.utils import cast_all_columns_to_numeric
import geopandas as gpd

test = gpd.geodataframe.GeoDataFrame()

out = cast_all_columns_to_numeric(test)
reveal_locals()
