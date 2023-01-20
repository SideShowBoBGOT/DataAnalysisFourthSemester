from sqlalchemy import Column, Integer, ForeignKey

from . import Base, ReprAttributesString, ID_STR, CASCADE
from my_models.main_models.transport_info import Transport
from my_models.main_models.communal_property import Property


class CommunalTransport(Base, ReprAttributesString):
    __tablename__ = 'communal_transport'

    def __init__(self, property_id: int, transport_id: int) -> None:
        self.property_id = property_id
        self.transport_id = transport_id

    id = Column(Integer, primary_key=True, autoincrement=True)
    property_id = Column(Integer, ForeignKey(
        Property.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)
    transport_id = Column(Integer, ForeignKey(
        Transport.__tablename__ + ID_STR, ondelete=CASCADE), nullable=False)