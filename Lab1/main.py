from my_etl_controller import get_main_engine, get_stage_engine, transform_data

def main():
    stage_engine = get_stage_engine(True)
    main_engine = get_main_engine(True)
    transform_data(stage_engine, main_engine)

if __name__ == '__main__':
    main()
