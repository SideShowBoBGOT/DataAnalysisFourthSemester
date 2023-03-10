from abc import ABC, abstractmethod
from typing import Any
import datetime as dt
import string
import random
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine
from my_models.main_models import Base

from my_etl_controller.connector import get_session


class TransLoader(ABC):

    def __init__(self, stage_engine: Engine, main_engine: Engine) -> None:
        self.stage_engine = stage_engine
        self.main_engine = main_engine
        self.stage_session = get_session(stage_engine)
        self.main_session = get_session(main_engine)
        self.meta = MetaData()
        self.start_date = dt.date(1863, 1, 1)
        self.end_date = dt.date(2023, 2, 27)

    @staticmethod
    def generate_date(start_date: dt.date, end_date: dt.date) -> dt.date:
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + dt.timedelta(days=random_number_of_days)
        return random_date

    @staticmethod
    def names_generator(size=6, chars=string.ascii_uppercase) -> str:
        return ''.join(random.choice(chars) for _ in range(size))

    def update_row_and_latest_id(self, value: str, d: dict[str, int], latest_id_ref: [int]) -> int:
        id = d.get(value)
        if id is None:
            id = latest_id_ref[0]
            d.update([(value, id)])
            latest_id_ref[0] = latest_id_ref[0] + 1
        return id

    def load_from_dict(self, d: dict[Any, Any], model: Base, autoincrement=True) -> None:
        if autoincrement:
            for name, _ in d.items():
                self.main_session.add(model(name))
        else:
            for name, id in d.items():
                self.main_session.add(model(id, name))
        self.main_session.commit()

    def load_from_list(self, l: list[tuple], model: Base) -> None:
        for row in l:
            self.main_session.add(model(*row))
        self.main_session.commit()

    @abstractmethod
    def models(self) -> list[Base]:
        pass

    def create_models(self):
        tables = list(map(lambda x: Base.metadata.tables[x.__tablename__], self.models()))
        Base.metadata.create_all(bind=self.main_engine, tables=tables)

    @abstractmethod
    def transform(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    def transload(self) -> None:
        self.create_models()
        self.transform()
        self.load()


from .au_transloader import AUTransLoader
from .cp_transloader import CPTransLoader
from .ti_transloader import TITransLoader
from .ct_transloader import CTTransLoader


def transform_data(stage_engine: Engine, main_engine: Engine) -> None:
    AUTransLoader(stage_engine, main_engine).transload()
    CPTransLoader(stage_engine, main_engine).transload()
    TITransLoader(stage_engine, main_engine).transload()
    CTTransLoader(stage_engine, main_engine).transload()
