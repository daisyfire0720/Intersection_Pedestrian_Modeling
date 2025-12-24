## import packages
import pandas as pd
from operator import add

zone_lst = ['bb','ff', 'lb', 'lf', 'll', 'rb', 'rf', 'rr']
ped_lst = ['_ped1', '_ped2', '_ped3', '_ped4', '_ped5']
var_lst = ['_x', '_y', '_vx', '_vy']
output_fp = r"C:\Users\Daisy\Downloads\GT Coursework\Research\Botao Coop\data\output\\"


def extract_area_importance(fn):
    input_name = fn + ' Permutation Importance Full.csv'
    df = pd.read_csv(output_fp + input_name)
    df_dict = dict(zip(df['feature_names'], df['feat_imp']))
    df_step_x = []
    for var in var_lst:
        columns_name = [item[1:] + var for item in ped_lst]
        df_var = pd.DataFrame(columns = columns_name, index = zone_lst)
        for ped in ped_lst:
            col_index = int(ped[-1])-1
            value_lst = []
            for zone in zone_lst:
                value_lst.append(df_dict[zone+ped+var])
            df_var.iloc[:, col_index] = value_lst
        df_step_x.append(df_var)
    df_lst_name = [item[1:] for item in var_lst]
    excel_name = fn + '_area.xlsx'
    writer = pd.ExcelWriter(output_fp + excel_name, engine = 'xlsxwriter')
    for idx, df in enumerate(df_step_x):
        df.to_excel(writer, sheet_name = df_lst_name[idx], index = True)
    writer.save()
    writer.close()
    
fn_lst = ['dest_vector_simple_step_x', 'dest_vector_simple_step_y']
for fn in fn_lst:
    extract_area_importance(fn)
