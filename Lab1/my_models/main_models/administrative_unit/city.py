from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .region import Region
from .previous_names_type import previous_names_type
import datetime as dt

class City(Base, ReprAttributesString):
    __tablename__ = 'city'

    def __init__(self, name: str, region_id: int, previous_names: list[tuple[str, dt.date]]) -> None:
        self.name = name
        self.region_id = region_id
        self.previous_names = previous_names

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    region_id = Column(Integer, ForeignKey(
        Region.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    previous_names = previous_names_type.copy()