from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .city_region import CityRegion


class Street(Base, ReprAttributesString):
    __tablename__ = 'street'

    def __init__(self, name: str, city_region_id: int) -> None:
        self.name = name
        self.city_region_id = city_region_id

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    city_region_id = Column(Integer, ForeignKey(
        CityRegion.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)