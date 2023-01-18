from sqlalchemy import Column, String, Integer, BIGINT, Date, Float
from ..mixins import ReprAttributesString
from . import Base

class TransportInfo(Base, ReprAttributesString):
    __tablename__ = 'transport_info'

    def __init__(self, reg_addr_koatuu: int, oper_code: int,
                oper_name: str, d_reg: str, dep_code: int, dep: str,
                brand: str, model: str, vin: str, make_year: int,
                color: str, kind: str, body: str, purpose: str, fuel: str,
                capacity: float, own_weight: float, total_weight: float, n_reg_new: str) -> None:
        self.reg_addr_koatuu = reg_addr_koatuu
        self.oper_code = oper_code
        self.oper_name = oper_name
        self.d_reg = d_reg
        self.dep_code = dep_code
        self.dep = dep
        self.brand = brand
        self.model = model
        self.vin = vin
        self.make_year = make_year
        self.color = color
        self.kind = kind
        self.body = body
        self.purpose = purpose
        self.fuel = fuel
        self.capacity = capacity
        self.own_weight = own_weight
        self.total_weight = total_weight
        self.n_reg_new = n_reg_new

    id = Column(Integer, primary_key=True, autoincrement=True)
    reg_addr_koatuu = Column(BIGINT)
    oper_code = Column(Integer)
    oper_name = Column(String)
    d_reg = Column(Date)
    dep_code = Column(Integer)
    dep = Column(String)
    brand = Column(String)
    model = Column(String)
    vin = Column(String)
    make_year = Column(Integer)
    color = Column(String)
    kind = Column(String)
    body = Column(String)
    purpose = Column(String)
    fuel = Column(String)
    capacity = Column(Float)
    own_weight = Column(Float)
    total_weight = Column(Float)
    n_reg_new = Column(String)