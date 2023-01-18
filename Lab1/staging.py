from my_etl_controller import get_main_engine, get_stage_engine, get_session, Extractor
from my_models import stage_models

def staging():
    stage_engine = get_stage_engine(True)
    stage_models.Base.metadata.create_all(stage_engine)
    stage_session = get_session(stage_engine)
    extractor = Extractor(stage_session)
    extractor.extract_data()

staging()