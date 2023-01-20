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
        self.property: list[tuple[int, int, int, float, float, int, float]] = []
        self.balance_keeper: dict[str, int] = {}
        self.component: dict[str, int] = {}
        self.structure: dict[str, int] = {}

    def transform(self) -> None:
        ids = {'bk_id': [1], 'struct_id': [1], 'comp_id': [1]}
        street_count = self.main_session.query(Street).count()
        for _, row in self.df_communal_property.iterrows():
            _, balance_keeper_name, _, structure_name, area, land_area, components_name, component_area, _ = row
            bk_id = self.update_row_and_latest_id(
                balance_keeper_name, self.balance_keeper, ids['bk_id'])
            struct_id = self.update_row_and_latest_id(
                structure_name, self.structure, ids['struct_id'])
            comp_id = self.update_row_and_latest_id(
                components_name, self.component, ids['comp_id'])
            street_id = np.random.randint(1, street_count)
            self.property.append(
                (bk_id, street_id, struct_id, area, land_area, comp_id, component_area))

    def load(self) -> None:
        self.load_from_dict(self.balance_keeper, BalanceKeeper)
        self.load_from_dict(self.component, Component)
        self.load_from_dict(self.structure, Structure)
        self.load_from_list(self.property, Property)
