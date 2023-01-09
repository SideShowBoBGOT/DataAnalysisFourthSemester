from datetime import date
from sqlalchemy import Column, String, Integer, Date
from . import Base, ReprAttributesString


class TransportInfoAndOwners(Base, ReprAttributesString):
    __tablename__ = 'transport_info_and_owners'

    def __int__(self, person: str, reg_addr_koatuu: str, oper_code: str,
                oper_name: str, d_reg: date, dep_code: int, dep: str,
                brand: str, model: str, vin: str, make_year: int,
                color: str, kind: str, body: str, purpose: str, fuel: str,
                capacity: int, own_weight: int, total_weight: int, n_reg_new: str) -> None:
        self.person = person
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
    person = Column(String)
    reg_addr_koatuu = Column(String)
    oper_code = Column(String)
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
    capacity = Column(Integer)
    own_weight = Column(Integer)
    total_weight = Column(Integer)
    n_reg_new = Column(String)