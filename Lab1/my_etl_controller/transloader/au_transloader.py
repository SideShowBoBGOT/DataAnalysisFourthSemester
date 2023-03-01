import pandas as pd
import datetime as dt
import random

from sqlalchemy.engine import Engine
from my_models.stage_models import AdministrativeUnit
from my_models.main_models import Base
from my_models.main_models.administrative_unit import Obl, Region, City, CityRegion, Street
from . import TransLoader


class AUTransLoader(TransLoader):
    def __init__(self, stage_engine: Engine, main_engine: Engine) -> None:
        TransLoader.__init__(self, stage_engine, main_engine)
        self.df_administrative_unit = pd.read_sql_table(
            AdministrativeUnit.__tablename__, self.stage_engine)
        self.table = {}
        self.max_names = 5
        self.max_length = 20

    def models(self) -> list[Base]:
        return [Obl, Region, City, CityRegion, Street]

    def transform(self) -> None:
        for i, row in self.df_administrative_unit.iterrows():
            _, obl_name, region_name, city_name, city_region_name, street_name = row
            regions = self.table.get(obl_name)
            if regions is None:
                regions = {}
                self.table.update([(obl_name, regions)])
            cities = regions.get(region_name)
            if cities is None:
                cities = {}
                regions.update([(region_name, cities)])
            city_regions = cities.get(city_name)
            if city_regions is None:
                city_regions = {}
                cities.update([(city_name, city_regions)])
            streets = city_regions.get(city_region_name)
            if streets is None:
                streets = []
                city_regions.update([(city_region_name, streets)])
            streets.append(street_name)

    def previous_names_generator(self) -> list[tuple[str, dt.date]]:
        previous_names = []
        number = random.randint(0, self.max_names)
        for _ in range(number):
            name = AUTransLoader.names_generator(self.max_length)
            rand_date = AUTransLoader.generate_date(self.start_date, self.end_date)
            previous_names.append((name, rand_date))
        return previous_names

    def load_obl(self):
        for obl_name, regions in self.table.items():
            self.main_session.add(Obl(obl_name, self.previous_names_generator()))
        self.main_session.commit()

    def load_region(self):
        obl_id = 1
        for obl_name, regions in self.table.items():
            for region_name, cities in regions.items():
                self.main_session.add(Region(region_name, obl_id, self.previous_names_generator()))
            obl_id = obl_id + 1
        self.main_session.commit()

    def load_city(self):
        region_id = 1
        for obl_name, regions in self.table.items():
            for region_name, cities in regions.items():
                for city_name, city_regions in cities.items():
                    self.main_session.add(City(city_name, region_id, self.previous_names_generator()))
                region_id = region_id + 1
        self.main_session.commit()

    def load_city_region(self):
        city_id = 1
        for obl_name, regions in self.table.items():
            for region_name, cities in regions.items():
                for city_name, city_regions in cities.items():
                    for city_region_name, streets in city_regions.items():
                        self.main_session.add(CityRegion(city_region_name, city_id, self.previous_names_generator()))
                    city_id = city_id + 1
        self.main_session.commit()

    def load_street(self):
        city_region_id = 1
        for obl_name, regions in self.table.items():
            for region_name, cities in regions.items():
                for city_name, city_regions in cities.items():
                    for city_region_name, streets in city_regions.items():
                        for street_name in streets:
                            self.main_session.add(Street(street_name, city_region_id, self.previous_names_generator()))
                        city_region_id = city_region_id + 1
        self.main_session.commit()

    def load(self):
        self.load_obl()
        self.load_region()
        self.load_city()
        self.load_city_region()
        self.load_street()
