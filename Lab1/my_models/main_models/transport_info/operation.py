from sqlalchemy import Column, String, Integer

from . import Base, ReprAttributesString


class Operation(Base, ReprAttributesString):
    __tablename__ = 'operation'

    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name

    id = Column(Integer, primary_key=True)
    name = Column(String)
