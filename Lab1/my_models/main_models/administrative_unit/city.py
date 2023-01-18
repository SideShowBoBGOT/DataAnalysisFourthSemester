from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .region import Region


class City(Base, ReprAttributesString):
    __tablename__ = 'city'

    def __init__(self, name: str, region_id: int) -> None:
        self.name = name
        self.region_id = region_id

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    region_id = Column(Integer, ForeignKey(
        Region.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)