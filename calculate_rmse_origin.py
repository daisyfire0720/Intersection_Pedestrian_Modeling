#%% import libraries
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fn_lst  = ['dest_polar_simple.csv', 'dest_polar.csv',
           'dest_vector_simple.csv','dest_vector.csv',
           'iner_polar_simple.csv', 'iner_polar.csv',
           'iner_vector_simple.csv', 'iner_vector.csv',
           'norm_polar_simple.csv', 'norm_polar.csv',
           'norm_vector_simple.csv', 'norm_vector.csv']
eval_dict = {}
output_fp = r"C:\Users\Daisy\Downloads\GT Coursework\Research\Botao Coop\data\output\\"

for fn in fn_lst:
    fn = fn[:-4]
    xls = pd.ExcelFile(output_fp + fn + '_lightgbm_output.xlsx')
    sheets = xls.sheet_names
    for sheet in sheets:
        df = xls.parse(sheet)
        if 'step_a' not in sheet:
            df['pred_diff'] = df.iloc[:, 0] - df.iloc[:, 1]
        if 'step_a' in sheet:
            df['pred_diff'] = df.apply(lambda x: x['step_a'] - x['pred_step_a'] if abs(x['step_a'] - x['pred_step_a']) <= 180 else 360 - abs(x['step_a'] - x['pred_step_a']), axis = 1)
        rmse = (df['pred_diff'] ** 2).mean() ** .5
        rmse_name = fn + '_' + sheet
        eval_dict[rmse_name] = rmse
        
eval_df = pd.DataFrame([eval_dict]).T.reset_index(drop = False)
eval_df.columns = ['name', 'value']
eval_df.to_csv(output_fp + 'performace_eval_original.csv', index = False)
        
