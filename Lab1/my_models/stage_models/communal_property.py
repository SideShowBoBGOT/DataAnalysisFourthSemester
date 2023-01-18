from sqlalchemy import Column, String, Integer, Float
from ..mixins import ReprAttributesString
from . import Base


class CommunalProperty(Base, ReprAttributesString):
    __tablename__ = 'communal_property'

    def __init__(self, balance_keeper: str, address: str, name: str,
                 area: float, land_area: float, components_name: str,
                 components_area: float, letters: str) -> None:
        self.balance_keeper = balance_keeper
        self.address = address
        self.name = name
        self.area = area
        self.land_area = land_area
        self.components_name = components_name
        self.components_area = components_area
        self.letters = letters

    id = Column(Integer, primary_key=True, autoincrement=True)
    balance_keeper = Column(String)
    address = Column(String)
    name = Column(String)
    area = Column(Float)
    land_area = Column(Float)
    components_name = Column(String)
    components_area = Column(Float)
    letters = Column(String)
