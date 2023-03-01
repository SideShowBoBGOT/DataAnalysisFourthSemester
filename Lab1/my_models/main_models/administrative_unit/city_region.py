from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .city import City

from .previous_names_type import previous_names_type
import datetime as dt

class CityRegion(Base, ReprAttributesString):
    __tablename__ = 'city_region'

    def __init__(self, name: str, city_id: int, previous_names: list[tuple[str, dt.date]]) -> None:
        self.name = name
        self.city_id = city_id
        self.previous_names = previous_names

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    city_id = Column(Integer, ForeignKey(
        City.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    previous_names = previous_names_type.copy()