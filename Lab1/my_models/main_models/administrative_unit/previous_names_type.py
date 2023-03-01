from sqlalchemy_utils import CompositeType
from sqlalchemy import Column, String, Date
from sqlalchemy.dialects.postgresql import ARRAY

name = 'previous_names'
types = [Column('name', String), Column('named_before', Date)]
composite_type = CompositeType(name, types)
previous_names_type = Column(ARRAY(composite_type, dimensions=1))
