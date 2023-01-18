from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .brand import Brand


class Model(Base, ReprAttributesString):
    __tablename__ = 'model'

    def __init__(self, name: str, brand_id: int) -> None:
        self.name = name
        self.brand_id = brand_id

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    brand_id = Column(Integer, ForeignKey(
        Brand.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
