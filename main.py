# -*- coding: utf-8 -*-


from fastapi import FastAPI
import pandas as pd
import pickle
from shap import TreeExplainer

app = FastAPI()

model = pickle.load(open("best_model_custom.pkl","rb"))

df = pd.read_csv('test_featureengineering.csv').iloc[:,1:]

shap_value = pickle.load(open("best_model_custom.pkl","rb"))

expected_value = pickle.load(open("expected_value.pkl","rb"))

@app.get("/{client}")
def get_score(client: int):
    df_score = df[df['SK_ID_CURR'] == int(client)].iloc[:,1:]
    
    return float(model.predict_proba(df_score)[:,1])

@app.get("/{client}/shap_value")
def get_shap_value(client: int):
    index = df.index[df['SK_ID_CURR'] == client].tolist()[0]
    
    return shap_value[index]

@app.get("/{client}/expected_value")
def get_shap_value(client: int):
    
    return expected_value