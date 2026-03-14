import dagster as dg
import sqlalchemy
from pydantic import PrivateAttr


class PostGISResource(dg.ConfigurableResource):
    host: str
    port: str
    user: str
    password: str
    db: str

    _engine: sqlalchemy.engine.Engine = PrivateAttr()

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:  # noqa: ARG002
        self._engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}",
        )

    def get_connection(self) -> sqlalchemy.engine.Connection:
        return self._engine.connect()
