from sqlalchemy import Column, Date, Integer, ForeignKey, Float
import datetime as dt

from . import Base, ReprAttributesString, ID_STR, CASCADE, Street
from .balance_keeper import BalanceKeeper
from .structure import Structure
from .component import Component


class Property(Base, ReprAttributesString):
    __tablename__ = 'property'

    def __init__(self, balance_keeper_id: int, street_id: int,
                 structure_id: int, area: Float, land_area: float,
                 component_id: int, component_area: float, reg_date: dt.date) -> None:
        self.balance_keeper_id = balance_keeper_id
        self.street_id = street_id
        self.structure_id = structure_id
        self.area = area
        self.land_area = land_area
        self.component_id = component_id
        self.component_area = component_area
        self.reg_date = reg_date

    id = Column(Integer, primary_key=True, autoincrement=True)
    reg_date = Column(Date)
    balance_keeper_id = Column(Integer, ForeignKey(
        BalanceKeeper.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    street_id = Column(Integer, ForeignKey(
        Street.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    structure_id = Column(Integer, ForeignKey(
        Structure.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    area = Column(Float)
    land_area = Column(Float)
    component_id = Column(Integer, ForeignKey(
        Component.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    component_area = Column(Float)
