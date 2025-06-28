import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
class PredictPipeline :
    def __init__(self):
        pass
# mapping all the values we give in html form to the backend
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
        self,
        age: float,
        bmi: float,
        sleep_duration: float,
        blood_pressure_systolic: float,
        blood_pressure_diastolic: float,
        heart_rate: float,
        skin_temperature: float,
        sensor_current_ua: float,

        sex: str,
        ethnicity: str,
        socioeconomic_status: str,
        physical_activity: str,
        alcohol_consumption: str,
        smoking_status: str,
        sleep_quality: str,
        stress_levels: str,
        family_history_diabetes: str,
        prediabetes: str,
        gestational_diabetes: str,
        hypertension: str,
        pcos: str,
        cardiovascular_disease: str,
        kidney_problems: str,
        medications: str,
        bmi_category: str
    ):
        self.age = age
        self.bmi = bmi
        self.sleep_duration = sleep_duration
        self.blood_pressure_systolic = blood_pressure_systolic
        self.blood_pressure_diastolic = blood_pressure_diastolic
        self.heart_rate = heart_rate
        self.skin_temperature = skin_temperature
        self.sensor_current_ua = sensor_current_ua

        self.sex = sex
        self.ethnicity = ethnicity
        self.socioeconomic_status = socioeconomic_status
        self.physical_activity = physical_activity
        self.alcohol_consumption = alcohol_consumption
        self.smoking_status = smoking_status
        self.sleep_quality = sleep_quality
        self.stress_levels = stress_levels
        self.family_history_diabetes = family_history_diabetes
        self.prediabetes = prediabetes
        self.gestational_diabetes = gestational_diabetes
        self.hypertension = hypertension
        self.pcos = pcos
        self.cardiovascular_disease = cardiovascular_disease
        self.kidney_problems = kidney_problems
        self.medications = medications
        self.bmi_category = bmi_category

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "bmi": [self.bmi],
                "sleep_duration": [self.sleep_duration],
                "blood_pressure_systolic": [self.blood_pressure_systolic],
                "blood_pressure_diastolic": [self.blood_pressure_diastolic],
                "heart_rate": [self.heart_rate],
                "skin_temperature": [self.skin_temperature],
                "sensor_current_ua": [self.sensor_current_ua],

                "sex": [self.sex],
                "ethnicity": [self.ethnicity],
                "socioeconomic_status": [self.socioeconomic_status],
                "physical_activity": [self.physical_activity],
                "alcohol_consumption": [self.alcohol_consumption],
                "smoking_status": [self.smoking_status],
                "sleep_quality": [self.sleep_quality],
                "stress_levels": [self.stress_levels],
                "family_history_diabetes": [self.family_history_diabetes],
                "prediabetes": [self.prediabetes],
                "gestational_diabetes": [self.gestational_diabetes],
                "hypertension": [self.hypertension],
                "pcos": [self.pcos],
                "cardiovascular_disease": [self.cardiovascular_disease],
                "kidney_problems": [self.kidney_problems],
                "medications": [self.medications],
                "bmi_category": [self.bmi_category],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

        except Exception as e:
            raise CustomException(e, sys)