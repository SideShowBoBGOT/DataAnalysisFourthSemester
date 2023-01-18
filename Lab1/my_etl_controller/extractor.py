import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from my_models.stage_models import AdministrativeUnit, CommunalProperty, TransportInfo


class Extractor:
    transport_info_file: str = 'data_sets/small_transport_info.csv'
    transport_info_delim: str = ';'
    communal_property_file: str = 'data_sets/communal_property.xls'
    administrative_unit_file: str = 'data_sets/administrative_unit.xml'

    def __init__(self, stage_session: Session):
        self.stage_session: Session = stage_session

    def extract_data(self):
        self.extract_communal_property_data()
        self.extract_transport_info_data()
        self.extract_administrative_unit_data()

    def extract_transport_info_data(self) -> None:
        data = pd.read_csv(self.transport_info_file,
                           delimiter=self.transport_info_delim)
        data = data.replace({np.nan: None})
        for el in data.values:
            self.stage_session.add(TransportInfo(*el[1:]))
        self.stage_session.commit()

    def extract_communal_property_data(self) -> None:
        data = pd.read_excel(self.communal_property_file)
        data['components_area'] = data['components_area'].str.replace(',', '.').astype(float)
        data['land_area'] = data['land_area'].str.replace(',', '.').astype(float)
        data['object_area'] = data['object_area'].str.replace(',', '.').astype(float)
        data = data.replace({np.nan: None})
        for el in data.values:
            self.stage_session.add(CommunalProperty(*el))
        self.stage_session.commit()

    def extract_administrative_unit_data(self) -> None:
        data = pd.read_xml(self.administrative_unit_file)
        data = data.replace({np.nan: None})
        for el in data.values:
            self.stage_session.add(AdministrativeUnit(*el))
        self.stage_session.commit()
