from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy_utils import database_exists, create_database, drop_database
import my_config


def get_engine(user: str, passwd: str, host: str, port: int, db: str, is_recreate_db: bool) -> Engine:
    url = f"postgresql://{user}:{passwd}@{host}:{port}/{db}"
    if is_recreate_db:
        if database_exists(url):
            drop_database(url)
        create_database(url)
    engine = create_engine(url, pool_size=50, echo=True)
    return engine


def check_connect_info(info: dict) -> None:
    if not all(key in my_config.keys for key in info.keys()):
        raise Exception('Bad info connect')


def get_stage_engine(is_recreate_db: bool) -> Engine:
    check_connect_info(my_config.stage_connector)
    info = my_config.stage_connector
    return get_engine(
        user=info['user'],
        passwd=info['passwd'],
        host=info['host'],
        port=info['port'],
        db=info['db'],
        is_recreate_db=is_recreate_db
    )


def get_main_engine(is_recreate_db: bool) -> Engine:
    check_connect_info(my_config.main_connector)
    info = my_config.main_connector
    return get_engine(
        user=info['user'],
        passwd=info['passwd'],
        host=info['host'],
        port=info['port'],
        db=info['db'],
        is_recreate_db=is_recreate_db
    )


def get_session(engine: Engine) -> Session:
    return sessionmaker(bind=engine)()
