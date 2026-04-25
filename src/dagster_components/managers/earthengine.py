import json

import dagster as dg
import ee

from dagster_components.managers.json import JSONManager


class EarthEngineManager(JSONManager):
    def handle_output(
        self,
        context: dg.OutputContext,
        obj: ee.image.Image | ee.geometry.Geometry,
    ) -> None:
        serialized = json.loads(obj.serialize())
        self._write_serialized_json(serialized, context)

    def load_input(
        self,
        context: dg.InputContext,
    ) -> ee.image.Image | ee.geometry.Geometry:
        serialized = self._read_serialized_json(context)
        deserialized = ee.deserializer.decode(serialized)

        if isinstance(deserialized, (ee.image.Image, ee.geometry.Geometry)):
            return deserialized

        err: str = f"Unsupported type: {type(deserialized)}"
        raise TypeError(err)
