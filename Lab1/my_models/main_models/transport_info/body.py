from sqlalchemy import Column, String, Integer

from . import Base, ReprAttributesString


class Body(Base, ReprAttributesString):
    __tablename__ = 'body'

    def __init__(self, name: str) -> None:
        self.name = name

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)