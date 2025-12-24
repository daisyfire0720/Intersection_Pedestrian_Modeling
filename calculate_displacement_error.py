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
        df_train_x = xls.parse('step_x_train')
        df_train_y = xls.parse('step_y_train')
        df_train = pd.concat([df_train_x, df_train_y], axis = 1)
        df_train['dis'] = df_train.apply(lambda x: math.sqrt((x['step_x'] - x['pred_step_x'])**2 + (x['step_y'] - x['pred_step_y'])**2), axis = 1)
        df_test_x = xls.parse('step_x_test')
        df_test_y = xls.parse('step_y_test')
        df_test = pd.concat([df_test_x, df_test_y], axis = 1)
        df_test['dis'] = df_test.apply(lambda x: math.sqrt((x['step_x'] - x['pred_step_x'])**2 + (x['step_y'] - x['pred_step_y'])**2), axis = 1)
        df_lst = [df_train, df_test]
        df_lst_name = ['train','test']
        writer = pd.ExcelWriter(output_fp + f'{fn}_displacement_error.xlsx', engine = 'xlsxwriter')
        for idx, df in enumerate(df_lst):
            df.to_excel(writer, sheet_name = df_lst_name[idx], index = False)
        writer.save()
        writer.close()
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
        df_train['dis'] = df_train.apply(lambda x: math.sqrt((x['step_x'] - x['pred_step_x'])**2 + (x['step_y'] - x['pred_step_y'])**2), axis = 1)
        df_test['step_x'] = df_test.apply(lambda x: x['step_l']*math.sin(x['step_a']/180*math.pi), axis = 1)
        df_test['step_y'] = df_test.apply(lambda x: x['step_l']*math.cos(x['step_a']/180*math.pi), axis = 1)
        df_test['pred_step_x'] = df_test.apply(lambda x: x['pred_step_l']*math.sin(x['pred_step_a']/180*math.pi), axis = 1)
        df_test['pred_step_y'] = df_test.apply(lambda x: x['pred_step_l']*math.cos(x['pred_step_a']/180*math.pi), axis = 1)
        df_test['dis'] = df_test.apply(lambda x: math.sqrt((x['step_x'] - x['pred_step_x'])**2 + (x['step_y'] - x['pred_step_y'])**2), axis = 1)
        df_lst = [df_train, df_test]
        df_lst_name = ['train','test']
        writer = pd.ExcelWriter(output_fp + f'{fn}_displacement_error.xlsx', engine = 'xlsxwriter')
        for idx, df in enumerate(df_lst):
            df.to_excel(writer, sheet_name = df_lst_name[idx], index = False)
        writer.save()
        writer.close()
        
        
        
        
        
        
        