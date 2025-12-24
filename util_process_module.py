#%% pre module
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#%% define module
class Util_process():
    def __init__(self, file_path, file_name):
        self.fp = file_path
        self.fn_lst = file_name
    def split_data(self, prop):
        for fn in self.fn_lst:
            df = pd.read_csv(self.fp + fn)
            df = df.drop(columns = ['id'])
            msk = np.random.rand(len(df)) < prop
            train = df[msk]
            test = df[~msk]
            train_name = fn[:-4] + '_train.csv'
            test_name = fn[:-4] + '_test.csv'
            train.to_csv(self.fp + train_name, index = False)
            test.to_csv(self.fp + test_name, index = False)

#%% run module
if __name__ == '__main__':
    file_path = r"C:\Users\Daisy\Downloads\GT Coursework\Research\Botao Coop\data\\"
    file_name = ['dest_polar.csv',
                 'dest_polar_simple.csv',
                 'dest_vector.csv',
                 'dest_vector_simple.csv',
                 'iner_polar.csv',
                 'iner_polar_simple.csv',
                 'iner_vector.csv',
                 'iner_vector_simple.csv',
                 'norm_polar.csv',
                 'norm_polar_simple.csv',
                 'norm_vector.csv',
                 'norm_vector_simple.csv']
    util = Util_process(file_path, file_name)
    util.split_data(0.8)
    

