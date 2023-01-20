from sqlalchemy import Column, String, Integer, Date, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from .operation import Operation
from .department import Department
from .model import Model
from .color import Color
from .kind import Kind
from .body import Body
from .purpose import Purpose
from .fuel import Fuel


class Transport(Base, ReprAttributesString):
    __tablename__ = 'transport'

    def __init__(self, reg_addr_koatuu: str, operation_id: int,
                 d_reg: str, department_id: int, model_id: int, vin: str,
                 make_year: int, color_id: int, kind_id: int, body_id: int,
                 purpose_id: int, fuel_id: int, capacity: int,
                 own_weight: int, total_weight: int, n_reg_new: str) -> None:
        self.reg_addr_koatuu = reg_addr_koatuu
        self.operation_id = operation_id
        self.d_reg = d_reg
        self.department_id = department_id
        self.model_id = model_id
        self.vin = vin
        self.make_year = make_year
        self.color_id = color_id
        self.kind_id = kind_id
        self.body_id = body_id
        self.purpose_id = purpose_id
        self.fuel_id = fuel_id
        self.capacity = capacity
        self.own_weight = own_weight
        self.total_weight = total_weight
        self.n_reg_new = n_reg_new

    id = Column(Integer, primary_key=True, autoincrement=True)
    reg_addr_koatuu = Column(String)
    operation_id = Column(Integer, ForeignKey(
        Operation.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    d_reg = Column(Date)
    department_id = Column(Integer, ForeignKey(
        Department.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    model_id = Column(Integer, ForeignKey(
        Model.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    vin = Column(String)
    make_year = Column(Integer)
    color_id = Column(Integer, ForeignKey(Color.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    kind_id = Column(Integer, ForeignKey(
        Kind.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    body_id = Column(Integer, ForeignKey(
        Body.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    purpose_id = Column(Integer, ForeignKey(
        Purpose.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    fuel_id = Column(Integer, ForeignKey(
        Fuel.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    capacity = Column(Integer)
    own_weight = Column(Integer)
    total_weight = Column(Integer)
    n_reg_new = Column(String)

