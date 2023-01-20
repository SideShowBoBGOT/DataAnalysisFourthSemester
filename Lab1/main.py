import pandas as pd

from my_etl_controller import get_main_engine, get_stage_engine, get_session, transform_data
from my_models import main_models


def main():
    stage_engine = get_stage_engine(False)
    main_engine = get_main_engine(False)
    transform_data(stage_engine, main_engine)

if __name__ == '__main__':
    main()
