import pandas as pd
import numpy as np
from sqlalchemy.engine import Engine
from my_models.stage_models import CommunalProperty
from my_models.main_models.communal_property import BalanceKeeper, Component, Property, Structure
from my_models.main_models.administrative_unit import Street
from . import TransLoader


class CPTransLoader(TransLoader):

    def __init__(self, stage_engine: Engine, main_engine: Engine) -> None:
        TransLoader.__init__(self, stage_engine, main_engine)
        self.df_communal_property = pd.read_sql_table(
            CommunalProperty.__tablename__, self.stage_engine)
        self.property: list[(int, int, int, int, float, float, float)] = []
        self.balance_keeper: dict[str, int] = {}
        self.street: dict[str, int] = {}
        self.component: dict[str, int] = {}
        self.structure: dict[str, int] = {}

    def transform(self):
        balance_keeper_id, component_id, structure_id  = (1, 1, 1)
        street_count = self.main_session.query(Street).count()
        for _, row in self.df_communal_property.iterrows():
            bk_id, comp_id, struct_id, strt_id = (0, 0, 0, 0)
            _, balance_keeper_name, _, name, area, land_area, components_name, components_area, _ = row
            if self.balance_keeper.get(balance_keeper_name) is None:
                bk_id = balance_keeper_id
                self.balance_keeper.update([(balance_keeper_name, balance_keeper_id)])
                balance_keeper_id = balance_keeper_id + 1


    def transload(self):
        pass
