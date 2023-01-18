from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
from .administrative_unit import AdministrativeUnit
from .communal_property import CommunalProperty
from .transport_info import TransportInfo
