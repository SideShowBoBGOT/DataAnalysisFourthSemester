from sqlalchemy import Column, String, Integer
from ..mixins import ReprAttributesString
from . import Base


class AdministrativeUnit(Base, ReprAttributesString):
    __tablename__ = 'administrative_unit'

    def __init__(self, obl: str, region: str, city: str, city_region: str, street: str) -> None:
        self.obl = obl
        self.region = region
        self.city = city
        self.city_region = city_region
        self.street = street

    id = Column(Integer, primary_key=True, autoincrement=True)
    obl = Column(String)
    region = Column(String)
    city = Column(String)
    city_region = Column(String)
    street = Column(String)
