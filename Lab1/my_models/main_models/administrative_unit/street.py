from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .city_region import CityRegion
from .previous_names_type import previous_names_type
import datetime as dt

class Street(Base, ReprAttributesString):
    __tablename__ = 'street'

    def __init__(self, name: str, city_region_id: int, previous_names: list[tuple[str, dt.date]]) -> None:
        self.name = name
        self.city_region_id = city_region_id
        self.previous_names = previous_names

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    city_region_id = Column(Integer, ForeignKey(
        CityRegion.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    previous_names = previous_names_type.copy()