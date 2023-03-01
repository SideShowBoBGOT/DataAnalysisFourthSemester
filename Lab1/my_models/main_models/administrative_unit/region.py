from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .obl import Obl
from .previous_names_type import previous_names_type
import datetime as dt

class Region(Base, ReprAttributesString):
    __tablename__ = 'region'

    def __init__(self, name: str, obl_id: int, previous_names: list[tuple[str, dt.date]]) -> None:
        self.name = name
        self.obl_id = obl_id
        self.previous_names = previous_names

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    obl_id = Column(Integer, ForeignKey(
        Obl.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    previous_names = previous_names_type.copy()