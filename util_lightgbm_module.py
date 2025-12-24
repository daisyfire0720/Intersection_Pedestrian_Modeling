#%% import libraries
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import sklearn
import eli5
import shap
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import *
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import statsmodels.api as sm
import os
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency

#%% define class
class Util_lightgbm():
    def __init__(self, input_fp, input_fn, output_fp):
        self.fp = input_fp
        self.fn = input_fn
        self.type = input_fn[:-4]
        self.fn_train = input_fn[:-4] + '_train.csv'
        self.fn_test = input_fn[:-4] + '_test.csv'
        self.output_fp = output_fp
    
    def _process_data(self):
        df_train = pd.read_csv(self.fp + self.fn_train)
        df_test = pd.read_csv(self.fp + self.fn_test)
        if 'polar' in self.fn:
            df_train_x = df_train.drop(columns = ['step_a', 'step_l', 'time'])
            df_train_angle = df_train['step_a']
            df_train_length = df_train['step_l']
            df_test_x = df_test.drop(columns = ['step_a', 'step_l', 'time'])
            df_test_angle = df_test['step_a']
            df_test_length = df_test['step_l']
            return df_train_x, df_train_angle, df_train_length, df_test_x, df_test_angle, df_test_length
        if 'vector' in self.fn:
            df_train_x = df_train.drop(columns = ['step_x', 'step_y', 'time'])
            df_train_length_x = df_train['step_x']
            df_train_length_y = df_train['step_y']
            df_test_x = df_test.drop(columns = ['step_x', 'step_y', 'time'])
            df_test_length_x = df_test['step_x']
            df_test_length_y = df_test['step_y']
            return df_train_x, df_train_length_x, df_train_length_y, df_test_x, df_test_length_x, df_test_length_y
    
    def _eval_metric_angle(self, y_true, y_pred):
        df_y = df_y = pd.concat([y_true, y_pred], axis = 1)
        df_y['pred_diff'] = df_y.apply(lambda x: x['step_a'] - x['pred_step_a'] if abs(x['step_a'] - x['pred_step_a']) <= 180 else 360 - abs(x['step_a'] - x['pred_step_a']), axis = 1)
        rmse = np.sqrt(sum(df_y['pred_diff']**2/len(df_y)))
        return rmse
    
    def _default_gbm(self, df_train_x, df_train_y, df_test_x, df_test_y):
        gbm = lgb.LGBMRegressor(objective = 'regression', verbose = False)
        gbm.fit(df_train_x, df_train_y, eval_metric = 'rmse')
        pred_name = 'pred_' + df_train_y.name
        train_y_true = df_train_y
        train_y_pred = pd.Series(gbm.predict(df_train_x)).rename(pred_name)
        df_train_pred = pd.concat([train_y_true, train_y_pred], axis = 1)
        test_y_true = df_test_y
        test_y_pred = pd.Series(gbm.predict(df_test_x)).rename(pred_name)
        df_test_pred = pd.concat([test_y_true, test_y_pred], axis = 1)
        if '_a' in train_y_true:
            print('Default train RMSE of {} is {:.4f}'.format(df_train_y.name, self._eval_metric_angle(train_y_true, train_y_pred)))
            print('Default test RMSE of {} is {:.4f}'.format(df_train_y.name, self._eval_metric_angle(test_y_true, test_y_pred)))
            eval_dict = {'train_rmse': self._eval_metric_angle(train_y_true, train_y_pred),
                         'test_rmse': self._eval_metric_angle(test_y_true, test_y_pred)}
        else:
            print('Default train RMSE of {} is {:.4f}'.format(df_train_y.name, mean_squared_error(train_y_true, train_y_pred)))
            print('Default test RMSE of {} is {:.4f}'.format(df_train_y.name, mean_squared_error(test_y_true, test_y_pred)))
            eval_dict = {'train_rmse': mean_squared_error(train_y_true, train_y_pred),
                         'test_rmse': mean_squared_error(test_y_true, test_y_pred)}
        fig_name = df_train_y.name + " LightGBM Feature Importance"
        lgb.plot_importance(gbm, importance_type = "gain", precision = 2,
                            figsize =(12,8), max_num_features = 10, title = fig_name)
        plt.show()
        return gbm, df_train_pred, df_test_pred, eval_dict
        
    def _optimized_gbm(self, df_train_x, df_train_y, df_test_x, df_test_y):
        lgb_train = lgb.Dataset(df_train_x, label = df_train_y, silent = True)
        fit_params = {'task': 'train',
                      'boosting_type': 'gbdt',
                      'objective': 'regression',
                      'metric': 'rmse',
                      "verbose": -1}
        estimator_cv_results = lgb.cv(fit_params,
                                      lgb_train,
                                      num_boost_round = 5000,
                                      nfold = 10,
                                      stratified = False,
                                      shuffle = False,
                                      metrics = 'rmse',
                                      early_stopping_rounds = 25,
                                      verbose_eval = False,
                                      seed = 42)
        param_test = {'min_child_samples':[20, 50, 64, 128, 256],
                      'min_child_weight':[1e-4, 1e-3, 1e-2, 1e-1],
                      'min_data_in_bin': [20, 50, 64, 128, 256],
                      'learning_rate':[0.05, 0.1, 0.15, 0.2, 0.25],
                      'num_leaves': sp_randint(6, 256),
                      'bagging_fraction':[0.75, 0.9, 0.95, 0.99],
                      'feature_fraction':[0.75, 0.9, 0.95, 0.99],
                      'reg_alpha': [1e-1, 1, 5, 10, 50],
                      'reg_lambda': [1e-1, 1, 5, 10, 50],
                      'max_depth': [6, 8, 10, 25, 50],
                      'max_bin': [32, 64, 128, 256],
                      'bagging_freq':[5, 10, 15, 20, 25],
                      'importance_type':['split', 'gain']}
        gbm = lgb.LGBMRegressor(n_estimators = len(estimator_cv_results['rmse-mean']),
                                n_jobs = None, **fit_params)
        rs_cv = RandomizedSearchCV(estimator = gbm,
                                   param_distributions = param_test,
                                   scoring = "neg_mean_squared_error",
                                   n_jobs = None,
                                   cv = 5,
                                   verbose = -1,
                                   n_iter = 100)
        rs_cv.fit(df_train_x, df_train_y)
        opti_params = rs_cv.best_params_
        opti_params["n_estimators"] = len(estimator_cv_results['rmse-mean'])
        opti_params["task"] = "train"
        opti_params["boosting_type"] = "gbdt"
        opti_params["objective"] = "regression"
        opti_params["metric"] = ["rmse"]
        print('Best score reached: {} with params: {} '.format(rs_cv.best_score_, rs_cv.best_params_))
        opti_params = rs_cv.best_params_
        param_fn = df_train_y.name + '_lgb_param.joblib'
        joblib.dump(opti_params, self.output_fp + param_fn)
        gbm_final = lgb.LGBMRegressor(**opti_params)
        gbm_final.fit(df_train_x, df_train_y, 
                      eval_set = [(df_train_x, df_train_y), (df_test_x, df_test_y)],
                      eval_metric = 'rmse',
                      early_stopping_rounds = 10,
                      verbose = False)
        pred_name = 'pred_' + df_train_y.name
        train_y_true_opti = df_train_y
        train_y_pred_opti = pd.Series(gbm_final.predict(df_train_x)).rename(pred_name)
        df_train_pred = pd.concat([train_y_true_opti, train_y_pred_opti], axis = 1)
        test_y_true_opti = df_test_y
        test_y_pred_opti = pd.Series(gbm_final.predict(df_test_x)).rename(pred_name)
        df_test_pred = pd.concat([test_y_true_opti, test_y_pred_opti], axis = 1)
        if '_a' in train_y_true_opti:
            print('Default train RMSE of {} is {:.4f}'.format(df_train_y.name, self._eval_metric_angle(train_y_true_opti, train_y_pred_opti)))
            print('Default test RMSE of {} is {:.4f}'.format(df_train_y.name, self._eval_metric_angle(test_y_true_opti, test_y_pred_opti)))
            eval_dict = {'train_rmse': eval_metric(train_y_true_opti, train_y_pred_opti),
                         'test_rmse': eval_metric(test_y_true_opti, test_y_pred_opti)}
        else:
            print('Default train RMSE of {} is {:.4f}'.format(df_train_y.name, mean_squared_error(train_y_true_opti, train_y_pred_opti)))
            print('Default test RMSE of {} is {:.4f}'.format(df_train_y.name, mean_squared_error(test_y_true_opti, test_y_pred_opti)))
            eval_dict = {'train_rmse':  mean_squared_error(train_y_true_opti, train_y_pred_opti),
                         'test_rmse':  mean_squared_error(test_y_true_opti, test_y_pred_opti)}
        
        return gbm_final, df_train_pred, df_test_pred, eval_dict
    
    def _feature_importance(self, gbm_model, df_test_x, df_test_y):
        plt.rcParams['font.family'] = 'Times New Roman'
        # lightgbm feature importance
        fig_name1 = self.type + '_' + df_test_y.name + " LightGBM Feature Importance"
        fig1 = lgb.plot_importance(gbm_model, importance_type = "gain", precision = 2,
                            figsize =(12,8), max_num_features = 10, title = fig_name1)
        plt.savefig(self.output_fp + fig_name1, dpi = 400)
        plt.show()
        # permutation importance
        perm = PermutationImportance(gbm_model, random_state = 42).fit(df_test_x, df_test_y)
        perm_weight = eli5.show_weights(perm, feature_names = df_test_x.columns.to_list())
        perm_result = pd.read_html(perm_weight.data)[0]
        perm_name = self.output_fp + self.type + '_' + df_test_y.name + ' Permutation Importance.csv'
        perm_result.to_csv(perm_name, index = False)
        df_perm = pd.DataFrame(dict(feature_names = df_test_x.columns.tolist(),
                                    feat_imp = perm.feature_importances_,
                                    std = perm.feature_importances_std_))
        df_perm_name = self.output_fp + self.type + '_' + df_test_y.name + ' Permutation Importance Full.csv'
        df_perm.to_csv(df_perm_name, index = False)
        # shap importance
        explainer = shap.Explainer(gbm_model)
        shap_values = explainer.shap_values(df_test_x)
        fig_name2 = self.type + '_' + df_test_y.name + " SHAP Feature Importance"
        shap.summary_plot(shap_values, df_test_x, max_display = 10, title = fig_name2, show = False)
        plt.savefig(self.output_fp + fig_name2, dpi = 400)
        plt.show()
        

    def main(self):
        df_train_x, df_train_y1, df_train_y2, df_test_x, df_test_y1, df_test_y2 = self._process_data()
        model1, train1, test1, eval_dict1 = self._optimized_gbm(df_train_x, df_train_y1, df_test_x, df_test_y1)
        self._feature_importance(model1, df_test_x, df_test_y1)
        model2, train2, test2, eval_dict2 = self._optimized_gbm(df_train_x, df_train_y2, df_test_x, df_test_y2)
        self._feature_importance(model2, df_test_x, df_test_y2)
        df_lst = [train1, test1, train2, test2]
        df_lst_name = [train1.columns[0] + '_train',
                        test1.columns[0] + '_test',
                        train2.columns[0] + '_train',
                        test2.columns[0] + '_test',]
        df_name = self.type
        # write output Excel file (ensure writer is properly closed using context manager)
        with pd.ExcelWriter(self.output_fp + f'{df_name}_lightgbm_output.xlsx', engine = 'xlsxwriter') as writer:
            for idx, df in enumerate(df_lst):
                df.to_excel(writer, sheet_name = df_lst_name[idx], index = False)
        eval_dict = {self.type + '_' + train1.columns[0]: eval_dict1,
                     self.type + '_' + train2.columns[0]: eval_dict2}
        return eval_dict
        

#%% run module
if __name__ == '__main__':
    fp = r"C:\Users\Daisy\Downloads\GT Coursework\Research\Botao Coop\data\\"
    output_fp = r"C:\Users\Daisy\Downloads\GT Coursework\Research\Botao Coop\data\output\\"
    fn_lst  = ['dest_polar_simple.csv', 'dest_polar.csv',
               'dest_vector_simple.csv','dest_vector.csv',
               'iner_polar_simple.csv', 'iner_polar.csv',
               'iner_vector_simple.csv', 'iner_vector.csv',
               'norm_polar_simple.csv', 'norm_polar.csv',
               'norm_vector_simple.csv', 'norm_vector.csv']
    fn_lst  = ['dest_polar_simple.csv', 'dest_vector_simple.csv']
    fn_lst = ['dest_vector_simple.csv']
    eval_dict = {}
    for fn in fn_lst:
        util_main = Util_lightgbm(fp, fn, output_fp)
        eval_fn = util_main.main()
        eval_dict = {**eval_dict, **eval_fn}
    eval_df = pd.DataFrame(eval_dict).T
    eval_df.to_excel(output_fp + 'performance_eval.xlsx')
