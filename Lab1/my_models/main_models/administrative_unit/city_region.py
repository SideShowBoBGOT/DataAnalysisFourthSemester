from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .city import City


class CityRegion(Base, ReprAttributesString):
    __tablename__ = 'city_region'

    def __init__(self, name: str, city_id: int) -> None:
        self.name = name
        self.city_id = city_id

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    city_id = Column(Integer, ForeignKey(
        City.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)