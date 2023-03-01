from sqlalchemy import Column, String, Integer

from . import Base, ReprAttributesString
from .previous_names_type import previous_names_type
import datetime as dt

class Obl(Base, ReprAttributesString):
    __tablename__ = 'obl'

    def __init__(self, name: str, previous_names: list[tuple[str, dt.date]]) -> None:
        self.name = name
        self.previous_names = previous_names

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    previous_names = previous_names_type.copy()