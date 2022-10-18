# -*- coding: utf-8 -*-


from fastapi import FastAPI
import pandas as pd
import pickle
from shap import TreeExplainer

app = FastAPI()

model = pickle.load(open("best_model_custom.pkl","rb"))

df = pd.read_csv('test_for_app.csv').iloc[:,1:]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df.iloc[:,1:])


@app.get("/{client}/score")
def get_score(client: int):
    df_score = df[df['SK_ID_CURR'] == int(client)].iloc[:,1:]
    
    return float(model.predict_proba(df_score)[:,1])

@app.get("/{client}/shap_value")
def get_shap_value(client: int):
    index = df.index[df['SK_ID_CURR'] == client].tolist()[0]
    
    return shap_values[0][index].tolist()

@app.get("/{client}/expected_value")
def get_shap_value(client: int):
    
    return explainer.expected_value[0]