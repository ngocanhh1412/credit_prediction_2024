# table.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

app = FastAPI()

with open('pipeline_model.pkl', 'rb') as file:
    pipeline_model = pickle.load(file)

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

@app.post("/bulk_predict/")
async def bulk_predict(file: UploadFile = File(...)):
    input_df = pd.read_csv(file.file)
    required_fields = ['name', 'member_id', 'term', 'int_rate', 'grade', 'emp_length', 'home_ownership',
                       'annual_inc', 'verification_status', 'purpose', 'dti', 'inq_last_6mths',
                       'revol_util', 'out_prncp', 'total_pymnt', 'total_rec_int',
                       'last_pymnt_amnt', 'tot_cur_bal', 'total_rev_hi_lim',
                       'earliest_cr_line', 'issue_d', 'last_pymnt_d', 'last_credit_pull_d']
    missing_fields = [field for field in required_fields if field not in input_df.columns]
    if missing_fields:
        return {"error": f"Missing required fields: {', '.join(missing_fields)}"}

    # Process the input data
    processed_df = dummy_creation(input_df[required_fields])
    scores = calculate_score(processed_df)
    input_df['score'] = scores

    # Save the result
    result_path = f"./results/table/{datetime.now().strftime('%Y%m%d%H%M%S')}_results.csv"
    input_df.to_csv(result_path, index=False)
    
    return input_df.to_dict(orient="records")

@app.get("/threshold")
async def get_threshold():
    with open('threshold.txt', 'r') as file:
        threshold = file.read().strip()
    return {"threshold": threshold}

