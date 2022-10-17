# -*- coding: utf-8 -*-


from fastapi import FastAPI
import pandas as pd
import pickle


app = FastAPI()

df = pd.read_csv('test_featureengineering.csv').iloc[:,1:]

model = pickle.load(open("best_model_custom.pkl","rb"))

@app.get("/{client}")
def get_score(client: int):

    return float(model.predict_proba(df[df['SK_ID_CURR'] == int(client)].iloc[:,1:])[:,1])
