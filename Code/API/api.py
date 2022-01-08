import uvicorn
import pickle
from fastapi import FastAPI
from pydantic import BaseModel


class Client(BaseModel):
    TOTALAREA_MODE: float
    AMT_CREDIT: float
    FLOORSMIN_MODE: float
    FLOORSMIN_MEDI: float
    NONLIVINGAREA_MODE: float
    LIVINGAREA_MODE: float
    ORGANIZATION_TYPE_Transporttype2: float
    NONLIVINGAREA_AVG: float
    BASEMENTAREA_AVG: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    BASEMENTAREA_MEDI: float
    FLOORSMAX_MEDI: float
    BASEMENTAREA_MEDI: float
    LIVINGAPARTMENTS_MODE: float
    AMT_REQ_CREDIT_BUREAU_MON: float
    LANDAREA_MODE: float
    FLOORSMAX_AVG: float
    FLOORSMIN_AVG: float
    BASEMENTAREA_MODE: float
    REGION_POPULATION_RELATIVE: float
    LIVINGAREA_AVG: float
    OBS_30_CNT_SOCIAL_CIRCLE: float
    OWN_CAR_AGE: float
    ORGANIZATION_TYPE_Security: float
    COMMONAREA_AVG: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    AMT_ANNUITY: float
    DAYS_REGISTRATION: float
    HOUR_APPR_PROCESS_START: float
    AMT_GOODS_PRICE: float
    COMMONAREA_MEDI: float
    DAYS_LAST_PHONE_CHANGE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    AMT_INCOME_TOTAL: float
    COMMONAREA_MODE: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    DAYS_ID_PUBLISH: float
    EXT_SOURCE_3: float


app = FastAPI()


with open("./LGBMClassifier.pkl", "rb") as f:

    model = pickle.load(f)

@app.get('/')

def index():

    return {'message': 'This is the homepage of the API '}

@app.post('/prediction/')

def get_solvablity_client(data: Client):

    received = data.dict()
    TOTALAREA_MODE = received['TOTALAREA_MODE']
    AMT_CREDIT = received['AMT_CREDIT']
    FLOORSMIN_MODE = received['FLOORSMIN_MODE']
    FLOORSMIN_MEDI = received['FLOORSMIN_MEDI']
    NONLIVINGAREA_MODE = received['NONLIVINGAREA_MODE']
    LIVINGAREA_MODE = received['LIVINGAREA_MODE']
    ORGANIZATION_TYPE_Transporttype2 = received['ORGANIZATION_TYPE_Transporttype2']
    NONLIVINGAREA_AVG = received['NONLIVINGAREA_AVG']
    BASEMENTAREA_AVG = received['BASEMENTAREA_AVG']
    OBS_60_CNT_SOCIAL_CIRCLE = received['OBS_60_CNT_SOCIAL_CIRCLE']
    AMT_REQ_CREDIT_BUREAU_QRT = received['AMT_REQ_CREDIT_BUREAU_QRT']
    BASEMENTAREA_MEDI = received['BASEMENTAREA_MEDI']
    FLOORSMAX_MEDI = received['FLOORSMAX_MEDI']
    LIVINGAPARTMENTS_MODE = received['LIVINGAPARTMENTS_MODE']
    AMT_REQ_CREDIT_BUREAU_MON = received['AMT_REQ_CREDIT_BUREAU_MON']
    LANDAREA_MODE = received['LANDAREA_MODE']
    FLOORSMAX_AVG = received['FLOORSMAX_AVG']
    FLOORSMIN_AVG = received['FLOORSMIN_AVG']
    BASEMENTAREA_MODE = received['BASEMENTAREA_MODE']
    LIVINGAREA_AVG = received['LIVINGAREA_AVG']
    REGION_POPULATION_RELATIVE = received['REGION_POPULATION_RELATIVE']
    OBS_30_CNT_SOCIAL_CIRCLE = received['OBS_30_CNT_SOCIAL_CIRCLE']
    OWN_CAR_AGE = received['OWN_CAR_AGE']
    ORGANIZATION_TYPE_Security = received['ORGANIZATION_TYPE_Security']
    COMMONAREA_AVG = received['COMMONAREA_AVG']
    AMT_REQ_CREDIT_BUREAU_YEAR = received['AMT_REQ_CREDIT_BUREAU_YEAR']
    AMT_ANNUITY = received['AMT_ANNUITY']
    DAYS_REGISTRATION = received['DAYS_REGISTRATION']
    HOUR_APPR_PROCESS_START = received['HOUR_APPR_PROCESS_START']
    AMT_GOODS_PRICE = received['AMT_GOODS_PRICE']
    COMMONAREA_MEDI = received['COMMONAREA_MEDI']
    DAYS_LAST_PHONE_CHANGE = received['DAYS_LAST_PHONE_CHANGE']
    DAYS_BIRTH = received['DAYS_BIRTH']
    DAYS_EMPLOYED = received['DAYS_EMPLOYED']
    COMMONAREA_MODE = received['COMMONAREA_MODE']
    EXT_SOURCE_1 = received['EXT_SOURCE_1']
    EXT_SOURCE_2 = received['EXT_SOURCE_2']
    AMT_INCOME_TOTAL = received['AMT_INCOME_TOTAL']
    DAYS_ID_PUBLISH = received['DAYS_ID_PUBLISH']
    EXT_SOURCE_3 = received['EXT_SOURCE_3']

    data = [[TOTALAREA_MODE, AMT_CREDIT, FLOORSMIN_MODE, FLOORSMIN_MEDI, NONLIVINGAREA_MODE, LIVINGAREA_MODE, ORGANIZATION_TYPE_Transporttype2, NONLIVINGAREA_AVG, BASEMENTAREA_AVG, OBS_60_CNT_SOCIAL_CIRCLE, AMT_REQ_CREDIT_BUREAU_QRT, BASEMENTAREA_MEDI, FLOORSMAX_MEDI, LIVINGAPARTMENTS_MODE, AMT_REQ_CREDIT_BUREAU_MON, LANDAREA_MODE, FLOORSMAX_AVG, FLOORSMIN_AVG, BASEMENTAREA_MODE, REGION_POPULATION_RELATIVE, LIVINGAREA_AVG, OBS_30_CNT_SOCIAL_CIRCLE, OWN_CAR_AGE,ORGANIZATION_TYPE_Security, COMMONAREA_AVG, AMT_REQ_CREDIT_BUREAU_YEAR, AMT_ANNUITY, DAYS_REGISTRATION, HOUR_APPR_PROCESS_START, AMT_GOODS_PRICE, COMMONAREA_MEDI, DAYS_LAST_PHONE_CHANGE, DAYS_BIRTH, DAYS_EMPLOYED, AMT_INCOME_TOTAL, COMMONAREA_MODE, EXT_SOURCE_1, EXT_SOURCE_2, DAYS_ID_PUBLISH,  EXT_SOURCE_3]]
    
    result_prob = model.predict_proba(data, pred_contrib=False)[0][0]
    print(result_prob)
    return {'prediction': result_prob}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
    # uvicorn.run(app)