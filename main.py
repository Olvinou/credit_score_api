# -*- coding: utf-8 -*-


from fastapi import FastAPI
import pandas as pd
import pickle


app = FastAPI()

model = pickle.load(open("best_model_custom.pkl","rb"))

@app.get("/{client}")
def get_score(client: int):
    df = pd.read_csv('test_featureengineering.csv').iloc[:,1:]
    df = df[df['SK_ID_CURR'] == int(client)].iloc[:,1:]
    return float(model.predict_proba(df)[:,1])
