import joblib

model_data = joblib.load('aquaintel_model.pkl')

model = model_data['model']
feature_names = model_data['feature_names']

from mltraining import aquaintel_predict_api

result = aquaintel_predict_api(
    temperature=25.0,
    rainfall=0.0,
    ph=7.1,
    dissolved_oxygen=6.8,
    current_water_level=5.5,
    village_name='Addateegala'
)
print(result)
