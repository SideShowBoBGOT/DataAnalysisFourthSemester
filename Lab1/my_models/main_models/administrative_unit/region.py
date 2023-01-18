from sqlalchemy import Column, String, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .obl import Obl


class Region(Base, ReprAttributesString):
    __tablename__ = 'region'

    def __init__(self, name: str, obl_id: int) -> None:
        self.name = name
        self.obl_id = obl_id

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    obl_id = Column(Integer, ForeignKey(
        Obl.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
