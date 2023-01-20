import pandas as pd
import numpy as np
from sqlalchemy.engine import Engine
from my_models.main_models import Base
from my_models.main_models.transport_info import Transport
from my_models.main_models.communal_property import Property
from my_models.main_models.communal_transport import CommunalTransport
from . import TransLoader


class CTTransLoader(TransLoader):

    def __init__(self, stage_engine: Engine, main_engine: Engine) -> None:
        TransLoader.__init__(self, stage_engine, main_engine)
        self.df_transport = pd.read_sql_table(
            Transport.__tablename__, self.main_engine)
        self.df_property = pd.read_sql_table(
            Property.__tablename__, self.main_engine)
        self.table: [(int, int)] = []

    def models(self) -> list[Base]:
        return [CommunalTransport]

    def transform(self) -> None:
        property_count = self.main_session.query(Property).count()
        for _, row in self.df_transport.iterrows():
            self.table.append((np.random.randint(1, property_count), row[0]))

    def load(self):
        self.load_from_list(self.table, CommunalTransport)
