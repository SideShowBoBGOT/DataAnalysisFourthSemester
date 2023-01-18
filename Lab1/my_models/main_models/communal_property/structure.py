from sqlalchemy import Column, String, Integer

from . import Base, ReprAttributesString


class Structure(Base, ReprAttributesString):
    __tablename__ = 'structure'

    def __init__(self, name: str) -> None:
        self.name = name

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)