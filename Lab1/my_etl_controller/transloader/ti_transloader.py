import pandas as pd
import numpy as np
from sqlalchemy.engine import Engine
from my_models.stage_models import TransportInfo
from my_models.main_models.transport_info import Body, Brand, Color, \
    Department, Fuel, Kind, Model, Operation, Purpose, Transport
from my_models.main_models import Base
from . import TransLoader


class TITransLoader(TransLoader):

    def __init__(self, stage_engine: Engine, main_engine: Engine) -> None:
        TransLoader.__init__(self, stage_engine, main_engine)
        self.df_transport_info = pd.read_sql_table(
            TransportInfo.__tablename__, self.stage_engine)
        self.transport: list[tuple] = []
        self.color: dict[str, int] = {}
        self.kind: dict[str, int] = {}
        self.fuel: dict[str, int] = {}
        self.body: dict[str, int] = {}
        self.purpose: dict[str, int] = {}
        self.reg_addr_koatuu: dict[int, int] = {}
        self.purpose: dict[str, int] = {}
        self.operation: dict[str, int] = {}
        self.department: dict[str, int] = {}
        self.purpose: dict[str, int] = {}
        # brand contains models
        self.brand: dict[str, dict[str, int]] = {}

    def transform_brand_model(self) -> None:
        model_id = 1
        for _, row in self.df_transport_info.iterrows():
            brand_name, model_name = (row[7], row[8])
            models = self.brand.get(brand_name)
            if models is None:
                models = {}
                self.brand.update([(brand_name, models)])
            if models.get(model_name) is None:
                models.update([(model_name, model_id)])
                model_id = model_id + 1

    def load_model(self) -> None:
        brand_id = 1
        for _, models in self.brand.items():
            for model_name, _ in models.items():
                self.main_session.add(Model(model_name, brand_id))
            brand_id = brand_id + 1
        self.main_session.commit()

    def nan_to_zero(self, number: int) -> int:
        result = number
        if np.isnan(result):
            result = 0
        return result

    def transform(self) -> None:
        self.transform_brand_model()
        ids = {'color_id': [1], 'kind_id': [1], 'fuel_id': [1],
               'body_id': [1], 'purpose_id': [1]}
        for _, row in self.df_transport_info.iterrows():
            _, reg_addr_koatuu, oper_code, oper_name, d_reg, \
                dep_code, dep_name, brand_name, model_name, vin, make_year, color_name, \
                kind_name, body_name, purpose_name, fuel_name, \
                capacity, own_weight, total_weight, n_reg_new = row
            color_id = self.update_row_and_latest_id(color_name, self.color, ids['color_id'])
            kind_id = self.update_row_and_latest_id(kind_name, self.kind, ids['kind_id'])
            fuel_id = self.update_row_and_latest_id(fuel_name, self.fuel, ids['fuel_id'])
            body_id = self.update_row_and_latest_id(body_name, self.body, ids['body_id'])
            purpose_id = self.update_row_and_latest_id(
                purpose_name, self.purpose, ids['purpose_id'])
            self.department.update([(dep_name, dep_code)])
            self.operation.update([(oper_name, oper_code)])
            model_id = self.brand[brand_name][model_name]
            capacity = self.nan_to_zero(capacity)
            own_weight = self.nan_to_zero(own_weight)
            total_weight = self.nan_to_zero(total_weight)
            self.transport.append(
                (reg_addr_koatuu, oper_code, d_reg, dep_code, model_id,
                 vin, make_year, color_id, kind_id, body_id, purpose_id,
                 fuel_id, capacity, own_weight, total_weight, n_reg_new))

    def load(self) -> None:
        self.load_from_dict(self.brand, Brand)
        self.load_model()
        self.load_from_dict(self.color, Color)
        self.load_from_dict(self.kind, Kind)
        self.load_from_dict(self.fuel, Fuel)
        self.load_from_dict(self.body, Body)
        self.load_from_dict(self.purpose, Purpose)
        self.load_from_dict(self.operation, Operation, autoincrement=False)
        self.load_from_dict(self.department, Department, autoincrement=False)
        self.load_from_list(self.transport, Transport)
