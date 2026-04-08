import numpy as np 
import joblib
import pandas as pd 
import shap
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all websites (development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class InputData(BaseModel):
    Age:int
    Gender:str
    Daily_Screen_time_Hours:float
    Social_Media_Hours:float 
    Online_Study_Hours:float
    Gaming_Hours:float 
    Sleep_Hours:float 
    Attendence_Percentage:float 
    Offline_Study_Hours:float 
    Previous_Sem_CGPA:float

Pipe = joblib.load(r"C:\FrostByte Project\Student CGPA Prediction ml model(MODIFIED).joblib")
explainer = joblib.load(r"C:\FrostByte Project\Shap explainer student cgpa predictor.pkl")
def Shap_explainations(Query_point):
    preprocessor = Pipe.named_steps["Scaler_Encod"]
    model = Pipe.named_steps["Model"]
    Query_point = preprocessor.transform(Query_point)
    shap_values = explainer(Query_point)
    values = shap_values.values[0]
    abs_values = np.abs(values)
    top_3_idx = np.argsort(abs_values)[-3:][::-1]
    feature_names = preprocessor.get_feature_names_out()
    Exp = {}
    for i in top_3_idx:
        clean_name = feature_names[i].split("__")[-1].replace("_" , " ").title()
        Exp[clean_name] = values[i]
    return Exp
@app.post("/predict")
def Predictor(Query_point:InputData):
    Query_Point = {"Age":Query_point.Age , "Gender":Query_point.Gender , "daily_screen_time_hours":Query_point.Daily_Screen_time_Hours , 
    "social_media_hours":Query_point.Social_Media_Hours , "online_study_hours":Query_point.Online_Study_Hours , 
    "gaming_hours":Query_point.Gaming_Hours , "sleep_hours":Query_point.Sleep_Hours , 
    "attendance_percentage":Query_point.Attendence_Percentage , 
    "offline_study_hours":Query_point.Offline_Study_Hours , "previous_sem_CGPA":Query_point.Previous_Sem_CGPA}
    Query_Point = pd.DataFrame([Query_Point])
    y_pred= Pipe.predict(Query_Point)
    reason = Shap_explainations(Query_Point)
    if y_pred[0] > 7.792000000000001:
        Remark = "Good"
    elif 6.54 <= y_pred[0] <= 7.792000000000001:
        Remark = "Medium" 
    else:
        Remark = "Low"
    formated_reasons = []
    for feature , value in reason.items():
        formated_reasons.append(
            {
                "feature":feature , 
                "impact":value , 
                "effect":"Increase" if value > 0 else "decrease"

            }
        )
    return {"Predicted CGPA: ":y_pred[0] , "Top 3 Reasons for this prediction: ":formated_reasons , "CGPA Level":Remark}

