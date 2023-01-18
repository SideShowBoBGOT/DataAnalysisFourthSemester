from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
from ..mixins import ReprAttributesString
from ..constants import ID_STR, CASCADE
from .administrative_unit import Obl, Region, City, CityRegion, Street
from .transport_info import Operation, Department, Brand, Model, Color, Kind, Body, Purpose, Fuel, Transport
from .communal_property import BalanceKeeper, Structure, Component, Property
