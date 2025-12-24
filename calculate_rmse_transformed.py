#%% import libraries
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
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
    if 'vector' in fn:
        for sheet in sheets:
            df = xls.parse(sheet)
            if 'step_a' not in sheet:
                df['pred_diff'] = df.iloc[:, 0] - df.iloc[:, 1]
            rmse = (df['pred_diff'] ** 2).mean() ** .5
            rmse_name = fn + '_' + sheet
            eval_dict[rmse_name] = rmse
    if 'polar' in fn:
        df_train_a = xls.parse('step_a_train')
        df_train_l = xls.parse('step_l_train')
        df_train = pd.concat([df_train_a, df_train_l], axis = 1)
        df_test_a = xls.parse('step_a_test')
        df_test_l = xls.parse('step_l_test')
        df_test = pd.concat([df_test_a, df_test_l], axis = 1)
        df_train['step_x'] = df_train.apply(lambda x: x['step_l']*math.sin(x['step_a']/180*math.pi), axis = 1)
        df_train['step_y'] = df_train.apply(lambda x: x['step_l']*math.cos(x['step_a']/180*math.pi), axis = 1)
        df_train['pred_step_x'] = df_train.apply(lambda x: x['pred_step_l']*math.sin(x['pred_step_a']/180*math.pi), axis = 1)
        df_train['pred_step_y'] = df_train.apply(lambda x: x['pred_step_l']*math.cos(x['pred_step_a']/180*math.pi), axis = 1)
        df_train['pred_diff_x'] = df_train.loc[:, 'step_x'] - df_train.loc[:, 'pred_step_x']
        df_train['pred_diff_y'] = df_train.loc[:, 'step_y'] - df_train.loc[:, 'pred_step_y']
        rmse_train_x = (df_train['pred_diff_x'] ** 2).mean() ** .5
        rmse_train_y = (df_train['pred_diff_y'] ** 2).mean() ** .5
        df_test['step_x'] = df_test.apply(lambda x: x['step_l']*math.sin(x['step_a']/180*math.pi), axis = 1)
        df_test['step_y'] = df_test.apply(lambda x: x['step_l']*math.cos(x['step_a']/180*math.pi), axis = 1)
        df_test['pred_step_x'] = df_test.apply(lambda x: x['pred_step_l']*math.sin(x['pred_step_a']/180*math.pi), axis = 1)
        df_test['pred_step_y'] = df_test.apply(lambda x: x['pred_step_l']*math.cos(x['pred_step_a']/180*math.pi), axis = 1)
        df_test['pred_diff_x'] = df_test.loc[:, 'step_x'] - df_test.loc[:, 'pred_step_x']
        df_test['pred_diff_y'] = df_test.loc[:, 'step_y'] - df_test.loc[:, 'pred_step_y']
        rmse_test_x = (df_test['pred_diff_x'] ** 2).mean() ** .5
        rmse_test_y = (df_test['pred_diff_y'] ** 2).mean() ** .5
        rmse_train_x_name = fn + '_' + 'step_x_train'
        rmse_train_y_name = fn + '_' + 'step_y_train'
        rmse_test_x_name = fn + '_' + 'step_x_test'
        rmse_test_y_name = fn + '_' + 'step_y_test'
        eval_dict[rmse_train_x_name] = rmse_train_x
        eval_dict[rmse_train_y_name] = rmse_train_y
        eval_dict[rmse_test_x_name] = rmse_test_x
        eval_dict[rmse_test_y_name] = rmse_test_y
        
eval_df = pd.DataFrame([eval_dict]).T.reset_index(drop = False)
eval_df.columns = ['name', 'value']
eval_df.to_csv(output_fp + 'performace_eval_transformed.csv', index = False)
        
        
        
        
        