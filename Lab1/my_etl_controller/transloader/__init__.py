from abc import ABC, abstractmethod
from sqlalchemy import MetaData, Table, Column
from sqlalchemy.engine import Engine
from ..connector import get_session

from my_models import main_models


class TransLoader(ABC):

    def __init__(self, stage_engine: Engine, main_engine: Engine) -> None:
        self.stage_engine = stage_engine
        self.main_engine = main_engine
        self.stage_session = get_session(stage_engine)
        self.main_session = get_session(main_engine)
        self.meta = MetaData()

    def populate_table(self, stage_column: Column, main_model: main_models.Base) -> None:
        for row in self.stage_session.query(stage_column).distinct():
            self.main_session.add(main_model(name=row[0]))
        self.main_session.commit()

    @abstractmethod
    def transload(self):
        pass


from .au_transloader import AUTransLoader
from .cp_transformer import CPTransLoader


def transform_data(stage_engine: Engine, main_engine: Engine) -> None:
    #AUTransLoader(stage_engine, main_engine).transload()
    CPTransLoader(stage_engine, main_engine).transload()
