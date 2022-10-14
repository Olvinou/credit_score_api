# -*- coding: utf-8 -*-


from fastapi import FastAPI
import pandas as pd
import pickle


app = FastAPI()

df = pd.read_csv('test_featureengineering.csv').iloc[:,1:]

model = pickle.load(open("best_model_custom.pkl","rb"))

scores = model.predict_proba(df.iloc[:,1:])[:,1]

df['SCORE'] = scores

@app.get("/{client}")
def get_score(client: int):
    return float(df['SCORE'][df['SK_ID_CURR'] == client])
