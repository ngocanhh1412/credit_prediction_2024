from woe import WoE_Binning
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

import csv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('pipeline_model.pkl', 'rb') as file:
    pipeline_model = pickle.load(file)
    
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

    
scorecard_scores = pd.read_csv('scorecard_scores.csv')

woe_transform = pipeline_model.named_steps['woe']
logistic_model= pipeline_model.named_steps['model']


# input -> json -> df
# input term: 36 months, 60 months
def handle_date_input(df):
    # chuyển date data về dạng yyyy-mm-01: ['earliest_cr_line', 'issue_d', 'last_pymnt_d', 'last_credit_pull_d']
    
    
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
    df[column] = pd.to_datetime(df[column], format = "%b-%y") 
    df['mths_since_' + column] = calculate_month_difference(df[column], today_date)
    df['mths_since_' + column] = df['mths_since_' + column].apply(lambda x: df['mths_since_' + column].max() if x < 0 else x)
    df.drop(columns = [column], inplace = True)

def handle_date(df):
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y') 
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'],format='%b-%y')
    df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'],format='%b-%y')
    
    date_columns(df, 'earliest_cr_line')
    date_columns(df, 'issue_d')
    date_columns(df, 'last_pymnt_d')
    date_columns(df, 'last_credit_pull_d')

    return df

def dummy_creation(df):
    df[features_to_keep]
    df = handle_date(df)
    df['term'] = pd.to_numeric(df['term'].str.replace(' months', ''))
    df_dummies = []
    for col in ['grade', 'home_ownership', 'verification_status', 'purpose']:
        df_dummies.append(pd.get_dummies(df[col], prefix = col, prefix_sep = ':'))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    df = df.reindex(labels = features,axis=1, fill_value=0)
    return df


def calculate_score(df):
    ref_categories = ['grade:G',  'home_ownership:MORTGAGE', 'verification_status:Not Verified', 'purpose:major_purch__car__home_impr', 'term:60', 'int_rate:>20.281', 
                  'annual_inc:>150K', 'dti:>35.191', 'inq_last_6mths:>4',  'revol_util:>1.0', 'out_prncp:>15,437', 'total_pymnt:>25,000', 'total_rec_int:>7,260', 
                  'total_rev_hi_lim:>79,780', 'mths_since_earliest_cr_line:>434',  'mths_since_issue_d:>122', 'mths_since_last_credit_pull_d:>75']
    
    df = woe_transform.fit_transform(df)
    df.insert(0, 'Intercept', 1)
    df = pd.concat([df, pd.DataFrame(dict.fromkeys(ref_categories, [0] * len(df)), 
                                                                 index = df.index)], axis = 1)
    
    print(df.shape)
    print(scorecard_scores.shape)
    scores = scorecard_scores.values.reshape(-1, 1)

    y_scores = df.dot(scores)
    
    return y_scores



if __name__ == '__main__':
    data = pd.read_csv('data/loan_data_2015.csv')
    
    data['target'] = np.where(data['loan_status'].isin(['Charged Off', 'Default',
                                                             'Late (31-120 days)',  'Late (16-30 days)']), 0, 1)
    data['target'].value_counts(normalize=True)
    data.drop(columns = ['loan_status'], inplace = True)
    
    loan_data = data[features_to_keep]
    
    sample_data = loan_data.sample(n=100, random_state=42) 
    
    transform_data = dummy_creation(sample_data)
    score = calculate_score(transform_data)

    score.to_csv("results.csv", index=False)
    print(score)