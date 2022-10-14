# -*- coding: utf-8 -*-


from fastapi import FastAPI
import pandas as pd
import pickle
import lightgbm

app = FastAPI()


model = pickle.load(open("best_model_custom.pkl","rb"))
df = []

@app.post("/receive_df")
def receive_df(df_in):
    df = df_in
    
    
@app.get("/{client}")
def get_score(client: int):
    return float(model.predict_proba(df[df['SK_ID_CURR'] == client].iloc[:,1:])[:,1])
