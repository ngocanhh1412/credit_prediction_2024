# form.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

app = FastAPI()

class CreditData(BaseModel):
    term: int
    int_rate: float
    grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    dti: float
    inq_last_6mths: int
    revol_util: float
    out_prncp: float
    total_pymnt: float
    total_rec_int: float
    last_pymnt_amnt: float
    tot_cur_bal: float
    total_rev_hi_lim: float
    earliest_cr_line: str
    issue_d: str
    last_pymnt_d: str
    last_credit_pull_d: str

with open('pipeline_model.pkl', 'rb') as file:
    pipeline_model = pickle.load(file)

# Ensure the file path is correct
scorecard_scores = pd.read_csv('scorecard_scores.csv')

def handle_date(df):
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y') 
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%y')
    df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format='%b-%y')
    
    date_columns(df, 'earliest_cr_line')
    date_columns(df, 'issue_d')
    date_columns(df, 'last_pymnt_d')
    date_columns(df, 'last_credit_pull_d')

    return df

def calculate_month_difference(date_series, reference_date):
    year_diff = reference_date.year - date_series.dt.year
    month_diff = reference_date.month - date_series.dt.month
    total_month_diff = year_diff * 12 + month_diff
    return total_month_diff

def date_columns(df, column):
    today_date = pd.to_datetime('2020-08-01')
    df[column] = pd.to_datetime(df[column], format="%b-%y")
    df['mths_since_' + column] = calculate_month_difference(df[column], today_date)
    df['mths_since_' + column] = df['mths_since_' + column].apply(lambda x: df['mths_since_' + column].max() if x < 0 else x)
    df.drop(columns=[column], inplace=True)

def dummy_creation(df):
    features_to_keep = ['term', 'int_rate', 'grade', 'emp_length', 'home_ownership',
                        'annual_inc', 'verification_status', 'purpose', 'dti', 'inq_last_6mths',
                        'revol_util', 'out_prncp', 'total_pymnt', 'total_rec_int',
                        'last_pymnt_amnt', 'tot_cur_bal', 'total_rev_hi_lim',
                        'earliest_cr_line', 'issue_d',
                        'last_pymnt_d', 'last_credit_pull_d']

    features = ['term', 'int_rate', 'grade', 'emp_length', 'home_ownership',
                'annual_inc', 'verification_status', 'purpose', 'dti', 'inq_last_6mths',
                'revol_util', 'out_prncp', 'total_pymnt', 'total_rec_int',
                'last_pymnt_amnt', 'tot_cur_bal', 'total_rev_hi_lim',
                'mths_since_earliest_cr_line', 'mths_since_issue_d',
                'mths_since_last_pymnt_d', 'mths_since_last_credit_pull_d', 'grade:A',
                'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
                'home_ownership:ANY', 'home_ownership:MORTGAGE', 'home_ownership:OWN',
                'home_ownership:RENT', 'verification_status:Not Verified',
                'verification_status:Source Verified', 'verification_status:Verified',
                'purpose:car', 'purpose:credit_card', 'purpose:debt_consolidation',
                'purpose:educational', 'purpose:home_improvement', 'purpose:house',
                'purpose:major_purchase', 'purpose:medical', 'purpose:moving',
                'purpose:other', 'purpose:renewable_energy', 'purpose:small_business',
                'purpose:vacation', 'purpose:wedding']

    df = df[features_to_keep]
    df = handle_date(df)
    # No need to convert term again
    df_dummies = []
    for col in ['grade', 'home_ownership', 'verification_status', 'purpose']:
        df_dummies.append(pd.get_dummies(df[col], prefix=col, prefix_sep=':'))
    df_dummies = pd.concat(df_dummies, axis=1)
    df = pd.concat([df, df_dummies], axis=1)
    df = df.reindex(labels=features, axis=1, fill_value=0)
    return df

def calculate_score(df):
    ref_categories = ['grade:G',  'home_ownership:MORTGAGE', 'verification_status:Not Verified', 'purpose:major_purch__car__home_impr', 'term:60', 'int_rate:>20.281', 
                      'annual_inc:>150K', 'dti:>35.191', 'inq_last_6mths:>4',  'revol_util:>1.0', 'out_prncp:>15,437', 'total_pymnt:>25,000', 'total_rec_int:>7,260', 
                      'total_rev_hi_lim:>79,780', 'mths_since_earliest_cr_line:>434',  'mths_since_issue_d:>122', 'mths_since_last_credit_pull_d:>75']
    
    df = pipeline_model.named_steps['woe'].transform(df)
    df.insert(0, 'Intercept', 1)
    df = pd.concat([df, pd.DataFrame(dict.fromkeys(ref_categories, [0] * len(df)), 
                                     index=df.index)], axis=1)
    
    scores = scorecard_scores.values.reshape(-1, 1)
    y_scores = df.dot(scores)
    
    return y_scores

@app.post("/predict")
async def predict(credit_data: CreditData):
    with open('threshold.txt', 'r') as file:
        threshold = float(file.read().strip())

    data_dict = credit_data.dict()
    df = pd.DataFrame([data_dict])
    df = dummy_creation(df)
    score = calculate_score(df)
    score_value = score[0][0]

    return {"score": score_value, "threshold": threshold}

# @app.get("/predict_by_id")
# async def predict_by_id():
#     print('abcde')
#     return 'test'
