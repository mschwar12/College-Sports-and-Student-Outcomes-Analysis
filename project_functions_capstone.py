#importing libraries
import pandas as pd
import numpy as np
from numpy import mean
import itertools
import warnings
import json
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
plt.style.use('ggplot')
import seaborn as sns

import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from yellowbrick.regressor import ResidualsPlot

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import KNNImputer
from sklearn import metrics

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

import pmdarima as pm
from pmdarima.arima import auto_arima
from pmdarima import model_selection
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose

import folium

def ms_evaluate(train_actual, train_predicted, test_actual, test_predicted):
    '''
    Courtesy of Lindsey Berlin
    
    Takes in both actual and predicted values, for the train and test set
    Then prints the scores based on those values
    
    Inputs:
    -------
    train_actual - actual target values for the train set
    train_predicted - predicted target values for the train set
    test_actual - actual target values for the test set
    test_predicted - predicted target values for the test set
    
    Outputs:
    -------
    
    train and test R2
    train and test MSE
    train and test RMSE
    '''
    print('Train R2:', r2_score(train_actual, train_predicted))
    print('Test R2:', r2_score(test_actual, test_predicted))
    print("*****")
    print('Train MSE:', mean_squared_error(train_actual, train_predicted))
    print('Test MSE:', mean_squared_error(test_actual, test_predicted))
    print("*****")
    print('Train RMSE:', mean_squared_error(train_actual, train_predicted, squared=False))
    print('Test RMSE:', mean_squared_error(test_actual, test_predicted, squared=False))

    
def ms_linear_regression(df, all_earn = None, dep = None, scale = None, id_x_cols = None, cols = None):
    '''
    Inputs:
    -------
    df - Pandas dataframe
    all_earn - if yes, indicates you want to provide a selected dependent variable
             - default is None - uses 6yr_mean_earnings as dependent  variable
    dep - if all_earn = 'yes', input desired dependent variable
    scale - if 'yes', function will scale X_vars
    id_x_cols - if 'yes', indicates you want to provide list of cols
              - default is None - uses all cols in df except for those institution name and earnings cols
    cols - list of cols - optional if you wish to use non-default column set
 
    Output: 
    -------
    multiple linear regression results (R2, RMSE) for train and test sets,
    residual scatter plot and histogram, list of variables and their coefficients
    '''
    
    if id_x_cols == 'yes':
        X_cols = cols
        X = df[X_cols]
    else:
        X_cols = [c for c in df.columns.to_list() if c not in ['INSTNM','6yr_mean_earnings', '6yr_female_mean_earnings',
                                                           '6yr_male_mean_earnings', '6yr_median_earnings',
                                                              '6yr_75th_pctile_earnings', '6yr_90th_pctile_earnings']]
        X = df[X_cols]
    
    num_cols = []
    for c in X.columns:
        if df[c].dtype in ['float64', 'int64']:
            num_cols.append(c)
    
    if all_earn == 'no':
        y = df[dep]
    else:
        y = df['6yr_mean_earnings']
    
#     X, y = np.arange(10).reshape((5, 2)), range(5)
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state=3)
    
    
#     num_transformer = Pipeline(steps=[
#     ('num_imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())])
    
#     preprocessor = ColumnTransformer(
#         transformers=[('num', num_transformer, num_cols)])

#     lr = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('regressor', LinearRegression())])

    imputer_simple = SimpleImputer(strategy = 'mean')
    
    ss = StandardScaler()
    
    X_train_imputed = imputer_simple.fit_transform(X_train)
    X_test_imputed = imputer_simple.transform(X_test)

    X_train_imputed = np.where(X_train_imputed == 0, 1, X_train_imputed)
    X_test_imputed = np.where(X_test_imputed == 0, 1, X_test_imputed)

    X_train_imputed_log = np.log(X_train_imputed)
    X_test_imputed_log = np.log(X_test_imputed)
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    
    if scale == 'yes':

        X_train_imputed_scaled = ss.fit_transform(X_train_imputed_log)
        X_test_imputed_scaled = ss.transform(X_test_imputed_log)

        lr = LinearRegression()
        lr.fit(X_train_imputed_scaled, y_train_log)
    
        #statsmodels for summary output
        X_train_imputed_scaled_df = pd.DataFrame(X_train_imputed_scaled, columns = X_cols).reset_index()
        X_train_imputed_scaled_df.drop(columns = 'index', inplace = True)
        X_train_imputed_scaled_sm_df = sm.add_constant(X_train_imputed_scaled_df)
        y_train_log_df = pd.DataFrame(y_train_log).reset_index()
        y_train_log_df.drop(columns = 'index', inplace = True)
        model = sm.OLS(y_train_log_df, X_train_imputed_scaled_sm_df)
        results = model.fit()
        summary = results.summary()

        y_pred_train = lr.predict(X_train_imputed_scaled)
        y_pred_test = lr.predict(X_test_imputed_scaled)
        
        visualizer = ResidualsPlot(lr, hist=False, qqplot=True)
    
        visualizer.fit(X_train_imputed_scaled, y_train_log)  # Fit the training data to the visualizer
        visualizer.score(X_test_imputed_scaled, y_test_log)  # Evaluate the model on the test data
        visualizer.show() 
        
    else:
        lr = LinearRegression()
        lr.fit(X_train_imputed_log, y_train_log)
    
        #statsmodels for summary output
        X_train_imputed_log_df = pd.DataFrame(X_train_imputed_log, columns = X_cols).reset_index()
        X_train_imputed_log_df.drop(columns = 'index', inplace = True)
        X_train_imputed_log_sm_df = sm.add_constant(X_train_imputed_log_df)
        y_train_log_df = pd.DataFrame(y_train_log).reset_index()
        y_train_log_df.drop(columns = 'index', inplace = True)
        model = sm.OLS(y_train_log_df, X_train_imputed_log_sm_df)
        results = model.fit()
        summary = results.summary()

        y_pred_train = lr.predict(X_train_imputed_log)
        y_pred_test = lr.predict(X_test_imputed_log)
        
        visualizer = ResidualsPlot(lr, hist=False, qqplot=True)
    
        visualizer.fit(X_train_imputed_log, y_train_log)  # Fit the training data to the visualizer
        visualizer.score(X_test_imputed_log, y_test_log)  # Evaluate the model on the test data
        visualizer.show() 
    
    y_pred_train_unlog = np.expm1(y_pred_train)
    y_pred_test_unlog = np.expm1(y_pred_test)
    
    y_train_unlog = np.expm1(y_train_log)
    y_test_unlog = np.expm1(y_test_log)

    coef = dict(zip(X.columns, lr.coef_))
    coef = pd.DataFrame.from_dict(coef, orient='index')
    coef.rename(columns={0: "coefficient"}, inplace=True)

    print(f"Train Score: {r2_score(y_train_log, y_pred_train)}")
    print(f"Test Score: {r2_score(y_test_log, y_pred_test)}")
    print('---')

    print('Train RMSE: ', np.sqrt(metrics.mean_squared_error(y_train_log, y_pred_train)))
    print('Test RMSE: ', np.sqrt(metrics.mean_squared_error(y_test_log, y_pred_test)))
    print('---')
    
    if np.isfinite(y_train_unlog).any() == False:
        pass
    else:
        print('Unlogged Train RMSE: ', np.sqrt(metrics.mean_squared_error(y_train_unlog, y_pred_train_unlog)))
        print('Unlogged Test RMSE: ', np.sqrt(metrics.mean_squared_error(y_test_unlog, y_pred_test_unlog)))
    print('---')
    
#     print('Intercept: ', lr.intercept_)
    
#     visualizer = ResidualsPlot(lr, hist=False, qqplot=True)
    
#     visualizer.fit(X_train_imputed_scaled, y_train_log)  # Fit the training data to the visualizer
#     visualizer.score(X_test_imputed_scaled, y_test_log)  # Evaluate the model on the test data
#     visualizer.show()   

    return summary 


def ms_eval_coefficients(model, column_names):
    '''
    Courtesy of Lindsey Berlin
    
    Prints an exploration of the coefficients
    
    Inputs:
    -------
    model - a fit linear model (sklearn)
    column_names - a list of feature names that matches the order passed into the model
    
    Outputs:
    -------
    coefs - a Series, sorted by coefficient value
    '''

    print("Total number of coefficients: ", len(model.coef_))
    print("Coefficients close to zero: ", sum(abs(model.coef_) < 10**(-10)))
    print(f"Intercept: {model.intercept_}")
    
    coefs = pd.Series(model.coef_, index= column_names)
    display(coefs.sort_values(ascending=False))
    return coefs.sort_values(ascending=False)

def ms_vif(df, id_cols = None, cols = None, nulls = None):
    '''
    Inputs: 
    -------
    df - Pandas dataframe
    nulls - if 'yes', function will use mean strategy to impute nulls
    
    Output: 
    -------
    A dataframe of VIF scores
    
    Variance inflation factor is a function from the statsmodels library.
    Rather than a correlation matrix, which tells you how correlated a pair of variables are,
    VIF is a wholistic metric that describes how correlated one feature is with all others.
    Anything over 5 is considered highly collinear.
    '''
    if id_cols == 'yes':
        X_cols = cols
    else:
        X_cols = [c for c in df.columns.to_list() if c not in ['INSTNM','6yr_mean_earnings', '6yr_female_mean_earnings',
                                                           '6yr_male_mean_earnings', '6yr_median_earnings']]
    X = df[X_cols]
    
    if nulls == 'yes':
        imputer_simple = SimpleImputer(strategy = 'mean')
        X_imputed = imputer_simple.fit_transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, columns = X.columns)
        
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X_imputed_df.values, i) for i in range(len(X_imputed_df.columns))]
        vif['features'] = X_imputed_df.columns
    else:    
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        vif['features'] = X.columns
    return vif




def ms_plot_univariate_panel(cols, data, func_plot, n_cols=2):

    '''
    Inputs: 
    -------
    cols - columns to use in plots
    data - Pandas dataframe
    func_plot - type of plot
    n_cols - number of columns wide to print subplots
    
    Output: 
    -------
    A grid of plots
    '''
    
    from math import ceil
    
    n_rows = ceil(len(cols) / n_cols)
    
    plt.figure(figsize=(7 * n_cols, 4 * n_rows))
    for idx, var in enumerate(cols, 1):
        plt.subplot(n_rows, n_cols, idx)
        func_plot(data[var])



def ms_ridge_regression(df, all_earn = None, dep = None, cols = None, col_list = None, scale = None, alpha = 10):
    '''
    Inputs:
    -------
    df - Pandas dataframe
    scale - if 'yes', function will scale X_vars
    col_list - if 'yes', indicates you want to provide list of cols
              - default is None - uses all cols in df except for those institution name and earnings cols
    cols - list of cols - optional if you wish to use non-default column set
    alpha - L2 penalty: penalizes sum of squared coefficients, default is 10
 
    Output: 
    -------
    ridge regression results (R2, RMSE) for train and test sets
    
    '''
    
    if col_list == 'yes':
        X_cols = cols
    else:
        X_cols = [c for c in df.columns.to_list() if c not in ['INSTNM','6yr_mean_earnings', 
                                                                               '6yr_female_mean_earnings',
                                                                               '6yr_male_mean_earnings', '6yr_median_earnings',
                                                                              '6yr_75th_pctile_earnings','6yr_90th_pctile_earnings']]

    X = df[X_cols]
    
    if all_earn == 'no':
        y = df[dep]
    else:
        y = df['6yr_mean_earnings']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state=3)


    imputer_simple = SimpleImputer(strategy = 'mean')

    ss = StandardScaler()

    X_train_imputed = imputer_simple.fit_transform(X_train)
    X_test_imputed = imputer_simple.transform(X_test)

    X_train_imputed = np.where(X_train_imputed == 0, 1, X_train_imputed)
    X_test_imputed = np.where(X_test_imputed == 0, 1, X_test_imputed)

    X_train_imputed_log = np.log(X_train_imputed)
    X_test_imputed_log = np.log(X_test_imputed)
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    
    if scale == 'yes':
        X_train_imputed_scaled = ss.fit_transform(X_train_imputed_log)
        X_test_imputed_scaled = ss.transform(X_test_imputed_log)

        alphas = list(range(1,21))

        ridge = Ridge(alpha)

        ridge.fit(X_train_imputed_scaled, y_train_log)

        r_train_preds = ridge.predict(X_train_imputed_scaled)
        r_test_preds = ridge.predict(X_test_imputed_scaled)


        # Evaluate
        print(ms_evaluate(y_train_log, r_train_preds, y_test_log, r_test_preds))
    else:
        alphas = list(range(1,21))

        ridge = Ridge(alpha)
        
        ridge.fit(X_train_imputed_log, y_train_log)

        # Predict
        r_train_preds = ridge.predict(X_train_imputed_log)
        r_test_preds = ridge.predict(X_test_imputed_log)

        #run cross val score for loop on train set, then test on test set
        #for loop for alpha score

        # Evaluate
        print(ms_evaluate(y_train_log, r_train_preds, y_test_log, r_test_preds))
    
    return ridge, X_cols



def ms_lasso_regression(df, all_earn = None, dep = None, cols = None, scale = None, col_list = None, alpha = .001 ):

    '''
    Inputs:
    -------
    df - Pandas dataframe
    scale - if 'yes', function will scale X_vars
    col_list - if 'yes', indicates you want to provide list of cols
              - default is None - uses all cols in df except for those institution name and earnings cols
    cols - list of cols - optional if you wish to use non-default column set
    alpha - L1 penalty: penalizes the sum of absolute values, default is .001
 
    Output: 
    -------
    lasso regression results (R2, RMSE) for train and test sets
    '''
    
    if col_list == 'yes':
        X_cols = cols
    else:
        X_cols = [c for c in df.columns.to_list() if c not in ['INSTNM','6yr_mean_earnings', 
                                                                               '6yr_female_mean_earnings',
                                                                               '6yr_male_mean_earnings', '6yr_median_earnings',
                                                                              '6yr_75th_pctile_earnings','6yr_90th_pctile_earnings']]

    X = df[X_cols]
    if all_earn == 'no':
        y = df[dep]
    else:
        y = df['6yr_mean_earnings']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state=3)


    imputer_simple = SimpleImputer(strategy = 'mean')

    ss = StandardScaler()

    X_train_imputed = imputer_simple.fit_transform(X_train)
    X_test_imputed = imputer_simple.transform(X_test)

    X_train_imputed = np.where(X_train_imputed == 0, 1, X_train_imputed)
    X_test_imputed = np.where(X_test_imputed == 0, 1, X_test_imputed)

    X_train_imputed_log = np.log(X_train_imputed)
    X_test_imputed_log = np.log(X_test_imputed)
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    if scale == 'yes':
        X_train_imputed_scaled = ss.fit_transform(X_train_imputed_log)
        X_test_imputed_scaled = ss.transform(X_test_imputed_log)

        alphas = list(range(1,21))

        lasso = Lasso(alpha)
        #grid = GridSearchCV(estimator = Ridge(), param_grid = dict(alpha = alphas))

        lasso.fit(X_train_imputed_scaled, y_train_log)
        #grid.fit(X_train_imputed_scaled, y_train_log)

        # print(grid.best_estimator_)
        # print(grid.best_score_)

        # Predict
        l_train_preds = lasso.predict(X_train_imputed_scaled)
        l_test_preds = lasso.predict(X_test_imputed_scaled)

        #run cross val score for loop on train set, then test on test set
        #for loop for alpha score

        # Evaluate
        print(ms_evaluate(y_train_log, l_train_preds, y_test_log, l_test_preds))
    else:
        alphas = list(range(1,21))

        lasso = Lasso(alpha)
        #grid = GridSearchCV(estimator = Lasso(), param_grid = dict(alpha = alphas))

        lasso.fit(X_train_imputed_log, y_train_log)
        #grid.fit(X_train_imputed_scaled, y_train_log)

        # print(grid.best_estimator_)
        # print(grid.best_score_)

        # Predict
        l_train_preds = lasso.predict(X_train_imputed_log)
        l_test_preds = lasso.predict(X_test_imputed_log)

        #run cross val score for loop on train set, then test on test set
        #for loop for alpha score

        # Evaluate
        print(ms_evaluate(y_train_log, l_train_preds, y_test_log, l_test_preds))
    
    return lasso, X_cols







def ms_elastic_net_cv(df, all_earn = None, dep = None, cols = None, col_list = None ):

    '''
    Inputs:
    -------
    df - Pandas dataframe
    col_list - if 'yes', indicates you want to provide list of cols
              - default is None - uses all cols in df except for those institution name and earnings cols
    cols - list of cols - optional if you wish to use non-default column set
 
    Output: 
    -------
    Elastic Net regression results (R2, RMSE) for train and test sets, x_cols used
    
    This function runs an Elastic Net cross validation model, where it tests a series
    of ratios (L1 penalty) and alphas (L2 penalty) to find best fitting model - operates like a grid search
    '''


    X_cols = [c for c in df.columns.to_list() if c not in ['INSTNM','6yr_mean_earnings', 
                                                                           '6yr_female_mean_earnings',
                                                                           '6yr_male_mean_earnings', '6yr_median_earnings',
                                                                          '6yr_75th_pctile_earnings','6yr_90th_pctile_earnings']]

    X = df[X_cols]
    
    if all_earn == 'no':
        y = df[dep]
    else:
        y = df['6yr_mean_earnings']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state=3)


    imputer_simple = SimpleImputer(strategy = 'mean')

    ss = StandardScaler()

    X_train_imputed = imputer_simple.fit_transform(X_train)
    X_test_imputed = imputer_simple.transform(X_test)

    X_train_imputed = np.where(X_train_imputed == 0, 1, X_train_imputed)
    X_test_imputed = np.where(X_test_imputed == 0, 1, X_test_imputed)

    X_train_imputed_log = np.log(X_train_imputed)
    X_test_imputed_log = np.log(X_test_imputed)
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    X_train_imputed_scaled = ss.fit_transform(X_train_imputed_log)
    X_test_imputed_scaled = ss.transform(X_test_imputed_log)

    #using kfolds for elastic net grid search (machine learning mastery)
    #this is doing my train test split, so let's fit this 
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    ratios = np.arange(0, 1, 0.01)
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]

    e_net_cv = ElasticNetCV(l1_ratio=ratios, alphas=alphas, n_jobs=-1)

    e_net_cv.fit(X_train_imputed_scaled, y_train_log)

    # Predict
    e_train_preds = e_net_cv.predict(X_train_imputed_scaled)
    e_test_preds = e_net_cv.predict(X_test_imputed_scaled)

    #run cross val score for loop on train set, then test on test set
    #for loop for alpha score

    # Evaluate
    ms_evaluate(y_train_log, e_train_preds, y_test_log, e_test_preds)
    print('alpha: %f' % e_net_cv.alpha_)
    print('l1_ratio_: %f' % e_net_cv.l1_ratio_)
    
    return e_net_cv, X_cols



def ms_elastic_net(df, cv = None, all_earn = None, dep = None, cols = None, col_list = None, params = None, alpha = 1.0, l1_ratio = 0.5, max_iter = 1000):

    '''
    Inputs:
    -------
    df - Pandas dataframe
    col_list - if 'yes', indicates you want to provide list of cols
              - default is None - uses all cols in df except for those institution name and earnings cols
    cols - list of cols - optional if you wish to use non-default column set
    l1_ratio - L1 penalty: penalizes the sum of absolute values, default is .001
    alpha - L2 penalty: penalizes the sum of absolute values, default is .1
    max_iter - maximum number of iterations, default is 1000
 
    Output: 
    -------
    Elastic Net regression results (R2, RMSE) for train and test sets, x_cols used
    
    This function runs a standard Elastic Net model, prints train and test scores,
    returns elastic net fitted model, x_cols used
    
    default values are from sklearn
    '''
    
    if col_list == 'yes':
        X_cols = cols
        X = df[X_cols]
    else:
        X_cols = [c for c in df.columns.to_list() if c not in ['INSTNM','6yr_mean_earnings', '6yr_female_mean_earnings',
                                                           '6yr_male_mean_earnings', '6yr_median_earnings',
                                                              '6yr_75th_pctile_earnings', '6yr_90th_pctile_earnings']]
        X = df[X_cols]
 
    
    if all_earn == 'no':
        y = df[dep]
    else:
        y = df['6yr_mean_earnings']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state=3)


    imputer_simple = SimpleImputer(strategy = 'mean')

    ss = StandardScaler()

    X_train_imputed = imputer_simple.fit_transform(X_train)
    X_test_imputed = imputer_simple.transform(X_test)

    X_train_imputed = np.where(X_train_imputed == 0, 1, X_train_imputed)
    X_test_imputed = np.where(X_test_imputed == 0, 1, X_test_imputed)

    X_train_imputed_log = np.log(X_train_imputed)
    X_test_imputed_log = np.log(X_test_imputed)
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    X_train_imputed_scaled = ss.fit_transform(X_train_imputed_log)
    X_test_imputed_scaled = ss.transform(X_test_imputed_log)
    
    ratios = np.arange(0, 1, 0.01)
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]

    if cv == 'yes':
        e_net_cv = ElasticNetCV(l1_ratio=ratios, alphas=alphas, n_jobs=-1)
        e_net_cv.fit(X_train_imputed_scaled, y_train_log)
        
        # Predict
        e_train_preds = e_net_cv.predict(X_train_imputed_scaled)
        e_test_preds = e_net_cv.predict(X_test_imputed_scaled)
        
        visualizer = ResidualsPlot(e_net_cv, hist=False, qqplot=True)
    
    else:
        e_net = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
        e_net.fit(X_train_imputed_scaled, y_train_log)
        
        # Predict
        e_train_preds = e_net.predict(X_train_imputed_scaled)
        e_test_preds = e_net.predict(X_test_imputed_scaled)
        
        visualizer = ResidualsPlot(e_net, hist=False, qqplot=True)
    
    #statsmodels for summary output
    X_train_imputed_scaled_df = pd.DataFrame(X_train_imputed_scaled, columns = X_cols).reset_index()
    X_train_imputed_scaled_df.drop(columns = 'index', inplace = True)
    X_train_imputed_scaled_sm_df = sm.add_constant(X_train_imputed_scaled_df)
    y_train_log_df = pd.DataFrame(y_train_log).reset_index()
    y_train_log_df.drop(columns = 'index', inplace = True)
    model = sm.OLS(y_train_log_df, X_train_imputed_scaled_sm_df)
    results_ols = model.fit()
    
    if params == 'yes':
        results_en = model.fit_regularized(method = 'elastic_net', alpha = alpha, L1_wt = l1_ratio, start_params = results_ols.params, refit=True)
        final = sm.regression.linear_model.OLSResults(model, results_en.params, model.normalized_cov_params)
        print('Alpha: ', alpha)
        print('L1 Ratio: ', l1_ratio)
    else:
        results_en = model.fit_regularized(method = 'elastic_net', alpha = e_net_cv.alpha_, L1_wt = e_net_cv.l1_ratio_, start_params = results_ols.params, refit=True)
        final = sm.regression.linear_model.OLSResults(model, results_en.params, model.normalized_cov_params)
        print('Alpha: ', e_net_cv.alpha_)
        print('L1 Ratio: ', e_net_cv.l1_ratio_)
    
    summary = final.summary()
    
    # Predict
    #e_train_preds = e_net_cv.predict(X_train_imputed_scaled)
    #e_test_preds = e_net_cv.predict(X_test_imputed_scaled)
        
    #visualizer = ResidualsPlot(e_net_cv, hist=False, qqplot=True)
    
    visualizer.fit(X_train_imputed_scaled, y_train_log)  # Fit the training data to the visualizer
    visualizer.score(X_test_imputed_scaled, y_test_log)  # Evaluate the model on the test data
    visualizer.show() 



    #run cross val score for loop on train set, then test on test set
    #for loop for alpha score

    # Evaluate
    ms_evaluate(y_train_log, e_train_preds, y_test_log, e_test_preds)
    
    if cv == 'yes':
        return summary, e_net_cv, X_cols
    else:
        return summary, e_net, X_cols
    #return e_net, 
    

def ms_repeated_kfolds(df, all_earn = None, dep = None):
    X_cols = [c for c in df.columns.to_list() if c not in ['INSTNM','6yr_mean_earnings', 
                                                                           '6yr_female_mean_earnings',
                                                                           '6yr_male_mean_earnings', '6yr_median_earnings',
                                                                          '6yr_75th_pctile_earnings','6yr_90th_pctile_earnings']]

    X = df[X_cols]
    
    if all_earn == 'no':
        y = df[dep]
    else:
        y = df['6yr_mean_earnings']

    # configure the cross-validation procedure
    cv_outer = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # enumerate splits
    train_results = list()
    test_results = list()
    
    imputer_simple = SimpleImputer(strategy = 'mean')

    ss = StandardScaler()

    for train, test in cv_outer.split(X):
        # split data
        X_train, X_test = X.iloc[train, :], X.iloc[test, :]
        y_train, y_test = y.iloc[train], y.iloc[test]
        
        #preprocess data
        X_train_imputed = imputer_simple.fit_transform(X_train)
        X_test_imputed = imputer_simple.transform(X_test)

        X_train_imputed = np.where(X_train_imputed == 0, 1, X_train_imputed)
        X_test_imputed = np.where(X_test_imputed == 0, 1, X_test_imputed)

        X_train_imputed_log = np.log(X_train_imputed)
        X_test_imputed_log = np.log(X_test_imputed)
        y_train_log = np.log(y_train)
        y_test_log = np.log(y_test)

        X_train_imputed_scaled = ss.fit_transform(X_train_imputed_log)
        X_test_imputed_scaled = ss.transform(X_test_imputed_log)
        
        # configure the cross-validation procedure
        cv_inner = RepeatedKFold(n_splits=3, n_repeats=3, random_state=1)

        # Use grid search to tune the parameters:

        parametersGrid = {"max_iter": [1, 5, 10, 100, 1000],
                          "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                          "l1_ratio": np.arange(0.0, 1.0, 0.1)}

        eNet = ElasticNet()
        grid = GridSearchCV(eNet, parametersGrid, scoring='r2', cv=10)
        
        # execute search
        result = grid.fit(X_train_imputed_scaled, y_train_log)

        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test_imputed_scaled)
        yhat_train = best_model.predict(X_train_imputed_scaled)

        # evaluate the model
        r2_train = r2_score(y_train_log, yhat_train)
        r2_test = r2_score(y_test_log, yhat)
        
        
        # store the result
        train_results.append(r2_train)
        test_results.append(r2_test)
        
        print(f"Train Score: {r2_score(y_train_log, yhat_train)}")
        print(f"Test Score: {r2_score(y_test_log, yhat)}")

        print('Train RMSE: ', np.sqrt(metrics.mean_squared_error(y_train_log, yhat_train)))
        print('Test RMSE: ', np.sqrt(metrics.mean_squared_error(y_test_log, yhat)))
        
        # report progress
        print('est=%.3f, cfg=%s' % (result.best_score_, result.best_params_))
        
        print('---')


    #summarize the estimated performance of the model
    print('Avg. train r2: %.3f, Avg. test r2: %.3f)' % (mean(r2_train), mean(r2_test)))

    print('---')

    print('Best Score=%.3f, Best Parameters=%s' % (result.best_score_, result.best_params_))
    
    return result.best_params_


def ms_sm_reg(summary):
    '''
    Inputs:
    ------- 
    summary - statsmodels summary results
    
    Output: 
    -------
    A new set of x_cols to be used in a dataframe with insignificant variables removed.
    Insignificance is determined by P-Value in a statsmodels regression model. Any variable with a P-value
    over .05 is excluded from the new dataframe.
    '''
#     outcome = 'price_log'
#     predictors = df.drop('price_log', axis=1)
#     pred_sum = '+'.join(predictors.columns)
#     formula = outcome + '~' + pred_sum
#     model = ols(formula=formula, data=df).fit()
    
#     summary = model.summary()
    p_table = summary.tables[1]
    p_table = pd.DataFrame(p_table.data)
    p_table.columns = p_table.iloc[0]
    p_table = p_table.drop(0)
    p_table = p_table.set_index(p_table.columns[0])
    p_table['P>|t|'] = p_table['P>|t|'].astype(float)
    x_cols = list(p_table[p_table['P>|t|'] < 0.05].index)
    x_cols.remove('const')
    print(len(p_table), len(x_cols))
    print(x_cols[:5])
    #new_df = df[x_cols]
    return x_cols
  
