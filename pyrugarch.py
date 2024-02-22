# created by maximo deleon
# thank you rugarch for existing <3 <- heart emoji
'''
MIT License

Copyright (c) 2024 pyrugarch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import datetime
import itertools
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from itertools import chain, combinations
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tsa.stattools import grangercausalitytests
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.conversion import localconverter
from tqdm import tqdm
#import openai
from statsmodels.stats.proportion import proportion_confint
import rpy2.rinterface as rinterface


#openai.api_key = #'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'
DECIMALS = 4
# rugarch imports
rugarch = importr('rugarch')
base = importr('base')
numpy2ri.activate()
pandas2ri.activate()



# General Research and Data Organization System

def powerset(s):
    x = len(s)
    pset = []
    for i in range(1 << x):
        pset.append([s[j] for j in range(x) if (i & (1 << j))]) 
    return pset


'''
pricing notes:

using Ada model: $0.0004/1K tokens

instructions: "The following table provides results for a GARCH(1,1) fit on different ETFs. Note that the endognous mean regressor is the log returns of the DBC ETF. Describe the results."
tokens: 44,                                     price = $0.0000176
table with 9 assets # of tokens: 1,733,         price = $0.0006932
estimated # of tokens for each asset: ~193,     price = $0.0000772

estimated price to get Ada Model to describe table with 9 assets = 0.0007112

x = (1 - 0.0000176)/0.0000772
x ~= 12953.139896373057 so it would cost approx $1 to write a description for 12,953 assets
'''

### batch analysis class ###
class Batch:
    def __init__(self, name, ModelClass, dataset = None, modelFitDataPath=None, modelFitS4Path=None, verbose=False) -> None:
        self.name = name
        self.ModelClass = ModelClass
        self.models = []
        self.model_names = []
        self.verbose = verbose
        self.dataset = dataset # './model_output/model_data/'

        self.hasExportedModelFitData = False
        self.hasExportedModelS4Data = False

        self.modelFitDataPath = '' if modelFitDataPath == None else modelFitDataPath
        self.modelFitS4Path = '' if modelFitS4Path == None else modelFitS4Path

        self.modelFitS4tag = ''
        self.modelFitDatatag = ''

        self.model_fit_table = None
        self.forecasts = []
        self.estimated_VaR = pd.DataFrame()
        self.backtest = pd.DataFrame()

    ### sets the export directories for either one of the paths used to export ###
    def setExportDirectories(self, modelFitDataPath=None, modelFitS4Path=None):
        self.modelFitDataPath = modelFitDataPath if modelFitDataPath != None else self.modelFitDataPath
        self.modelFitS4Path = modelFitS4Path if modelFitS4Path != None else self.modelFitS4Path

    ### add a model ###
    def addModel(self, model , verbose=False) -> None:
        if type(model) == self.ModelClass:
            self.models.append(model)
            self.model_names.append(model.id)
            if self.verbose == True or verbose == True:
                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {self.name} | Added model with id: {model.id}')
            
        else:
            print(f'type(model) != {self.ModelClass}')

    ### add a list of models ###
    def addModels(self, models, verbose=False) -> None:

        initial_model_count = len(self.models)
        for model in models:
            self.addModel(model)
        
        if self.verbose == True or verbose == True:
            dM = len(self.models) - initial_model_count
            ms = 'models' if dM > 1 else 'model'
            print(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {self.name} | Added {dM} {ms}')
    
    ### exports all S4 fit data ###
    def exportS4(self, tag=None, directory=None, verbose=False) -> None:
        
        tag = self.name if tag == None else tag
        self.modelFitS4Path = directory
        self.modelFitS4tag = tag

        for model in self.models:
            model.exportFitS4(tag=tag,directory=directory)
            if verbose == True or self.verbose == True:
                print(f'Exported {model.id}_{tag} fit summary to {directory}{model.id}_{tag}.txt')

        self.hasExportedModelS4Data = True

    ### exports all fit data ###
    def exportFitData(self, tag=None, directory=None, verbose=False) -> None:
        
        tag = self.name if tag == None else tag

        self.modelFitDataPath = directory
        self.modelFitDatatag = tag
        for model in self.models:
            model.exportFitData(tag=tag,directory=directory)

            if verbose == True or self.verbose == True:
                print(f'Exported {model.id}_{tag} fit data to {directory}{model.id}_{tag}.xlsx')
            
        self.hasExportedModelFitData = True
        
    ### fits all models loaded ###
    def batchFit(self, rugarch_fit=True, dataset=None, export=False, verbose=False) -> None:
        
        if self.verbose == True or verbose == True:
            ms = 'models' if len(self.models) > 1 else 'model'
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {self.name} | Starting batch fit on {len(self.models)} {ms}')
        
        if dataset is None and self.dataset is None:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {self.name} | Cannot fit model with no dataset loaded')
            return

        elif dataset is None and isinstance(self.dataset, pd.DataFrame):
            selected_dataset = self.dataset

        elif isinstance(dataset, pd.DataFrame) and self.dataset is None:
            selected_dataset = dataset

        elif isinstance(dataset, pd.DataFrame) and isinstance(self.dataset, pd.DataFrame):
            selected_dataset = dataset

        else:
            print('CRITICAL UH OH')
            return

        bad_fits = 0
        for model in self.models:
            try:
                if rugarch_fit == True:
                    model.fit(selected_dataset)
                else:
                    model.fitOLS(selected_dataset)

            except Exception as e:
                bad_fits += 1
                print(f'Failed to fit model! id:{model.id}')
                print(e)
    

        if self.verbose == True or verbose == True:
            ms = 'models' if len(self.models)-bad_fits > 1 else 'model'
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {self.name} | Completed batch fit on {len(self.models)-bad_fits} {ms}')
    
        else: pass

    ### perform rolling fit on models in batch
    def batchRollingFit(self, dataset=None, distribution_model='std', nStart=100, refit=10 ,n_ahead=1, window_type='moving', verbose=False):

        if self.verbose == True or verbose == True:
            ms = 'models' if len(self.models) > 1 else 'model'
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {self.name} | Starting batch fit on {len(self.models)} {ms}')
        
        if dataset is None and self.dataset is None:
            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {self.name} | Cannot fit model with no dataset loaded')
            return

        elif dataset is None and isinstance(self.dataset, pd.DataFrame):
            selected_dataset = self.dataset

        elif isinstance(dataset, pd.DataFrame) and self.dataset is None:
            selected_dataset = dataset

        elif isinstance(dataset, pd.DataFrame) and isinstance(self.dataset, pd.DataFrame):
            selected_dataset = dataset


        bad_fits = 0
        for model in self.models:
            try: #df, endog, mean_exog, variance_exog, garch_model, arma_order, garch_order, distribution_model='std', nStart=100, refit=10, window_type='moving'
                model.rollingFit(data=selected_dataset, nStart=nStart, refit=refit, n_ahead=n_ahead, window_type=window_type)

            except Exception as e:
                bad_fits += 1
                print(f'Failed to perform rolling fit with model! id:{model.id}')
                print(e)
    
    ### batch historical VaR with rolling option and confidence level
    def batchHistoricalVaR(self, rolling=False, alpha=0.05, use_robust=True):
        
        bad_fits = 0
        estimated_var_dict = {}
        for model in self.models:
            try: #df, endog, mean_exog, variance_exog, garch_model, arma_order, garch_order, distribution_model='std', nStart=100, refit=10, window_type='moving'
                estimated_var_dict[model.id] = model.historicalVaR(rolling=rolling, alpha=alpha, use_robust=use_robust) # var output not used
            
            except Exception as e:
                bad_fits += 1
                print(f'Failed to generate histocial VaR! id:{model.id}')
                print(e)

        

        self.estimated_VaR = pd.concat(estimated_var_dict,axis=1) #pd.concat(estimated_var_dict,axis=1)

    ### batch forecast
    def batchForecast(self, n_ahead=1):
        bad_fits = 0
        forecasts = {}
        for model in self.models:
            try:
                forecasts[model.id] = model.forecast(n_ahead)

            except Exception as e:
                bad_fits += 1
                print(f'Failed to fit model! id:{model.id}')
                print(e)

        self.forecasts = pd.concat(forecasts,axis=1)
    
    # Counts the number of times the VaR model is exceeded by the returns of the series
    # generates a binomial distribution as a means to determimne VaR model efficacy as argued in Kupiec (1995), the failure number follows a binomial distribution, B(T, p).
    def batchEvaluateVaR(self, rolling=False, model_alpha=0.05, binomial_alpha=0.05, use_robust=True):

        bad_fits = 0
        var_performance = {}
        for model in self.models:
            try: 
                var_performance[model.id] = model.evaluateVaR(rolling=rolling, model_alpha=model_alpha, binomial_alpha=binomial_alpha, use_robust=use_robust) # var output not used

            except Exception as e:
                bad_fits += 1
                print(f'Failed to backtest VaR model! id:{model.id}')
                print(e)
        
        var_performance = pd.DataFrame(var_performance)
        if rolling == True:
            self.rolling_var_performance = var_performance
        else:
            self.fixed_var_performance = var_performance

    ### does export on model fit data and S4 data ###
    def export(self, verbose=False):
        if self.modelFitDataPath != '' and self.modelFitDataPath != '':
            self.exportFitData(directory=self.modelFitDataPath, verbose=verbose)
            self.exportS4(directory=self.modelFitS4Path, verbose=verbose)
        else: pass

    ### returns a table in a format for the language model to use ###
    def modelFitTable(self, for_latex=False) -> pd.DataFrame:
        if self.hasExportedModelFitData == False or self.modelFitDataPath == '':
            table = pd.DataFrame()
        else:
            mxreg_input = {}
            vxreg_input = {}

            # this is garbage. Absolute garbage. This is meant to be a get it done part of the code
            # somone in the future fix this pls
            for i,model in enumerate(self.models):

                if model.mean_exog != None:
                    mean_regressor_keys = [f'mxreg{1+i}' for i in range(len(model.mean_exog))]
                    mean_regressor_values = [v for v in model.mean_exog]
                    mxreg_input[str([self.model_names[i]])] = dict(zip(mean_regressor_keys, mean_regressor_values))

                if model.variance_exog != None:
                    variance_regressor_keys = [f'vxreg{1+i}' for i in range(len(model.variance_exog))]
                    variance_regressor_values = [v for v in model.variance_exog]
                    vxreg_input[str([self.model_names[i]])] = dict(zip(variance_regressor_keys, variance_regressor_values))

            table = buildTableFromOutputData(target_directory= self.modelFitDataPath, 
                                                tagged_with= f'_{self.modelFitDatatag}.xlsx', 
                                                symbols= self.model_names.copy(), 
                                                mxregs=mxreg_input, 
                                                vxregs= vxreg_input, 
                                                forLanguageModel= not for_latex)
        
        return table


### simple class to store the model parameters ###
class RugarchModel:
    def __init__(self, id, endog, mean_exog, variance_exog, garch_model, arma_order, garch_order, distribution_model='std'):
        self.id, self.endog, self.mean_exog, self.variance_exog, self.garch_model, self.arma_order, self.garch_order, self.distribution_model \
            = id, endog, mean_exog, variance_exog, garch_model, arma_order, garch_order, distribution_model


        # statistical model results
        self.fit_data, self.fitS4 = None, None
        self.rolling_fit_data, self.rolling_fitS4 = None, None
        self.ols_model, self.ols_fit = None, None
        self.dataset = None
        self.roll_start_idx = 0
        self.historical_var_estimate = None
        self.fixed_var_performance_data = {}
        self.rolling_var_performance_data = {}

    ### standard GARCH fit to data ###
    def fit(self, data):
        self.fit_data, self.fitS4 = rugarchFit(data, self.endog, self.mean_exog, self.variance_exog, self.garch_model, self.arma_order, self.garch_order, self.distribution_model)
        self.dataset = data
    ### rolling GARCH fit to data ###
    def rollingFit(self, data, nStart=100, refit=10, n_ahead=1, window_type='moving'):
        self.rolling_fit_data, self.rolling_fitS4 = rugarchRollFit(df=data, 
                                                                   endog=self.endog, 
                                                                   mean_exog=self.mean_exog, 
                                                                   variance_exog=self.variance_exog, 
                                                                   garch_model=self.garch_model, 
                                                                   arma_order=self.arma_order, 
                                                                   garch_order=self.garch_order, 
                                                                   distribution_model=self.distribution_model, 
                                                                   nStart=nStart,
                                                                   refit=refit, 
                                                                   n_ahead=n_ahead, 
                                                                   window_type=window_type)
        self.dataset = data
        self.roll_start_idx = nStart

    ### forecast using fit
    def forecast(self, n_ahead, dataset=None):
        global_env = robjects.globalenv
        dataset = self.dataset if dataset == None else dataset

        global_env['forecast'] = rugarch.ugarchforecast(self.fitS4,n_ahead=n_ahead,data=dataset)
        
        forecast = robjects.conversion.rpy2py(r('sigma(forecast)'))
        dataset.index = pd.to_datetime(dataset.index)
        base = dataset.index[-1]
        date_list = [base + datetime.timedelta(days=x) for x in range(1, n_ahead+1)]
        forecast = pd.DataFrame(forecast, index=pd.to_datetime(date_list),columns=['SigmaFcst'])

        return forecast

    def historicalVaR(self, rolling=False, alpha=0.05, use_robust=True):
        
        # ONLY HAS SUPPORT FOR T DISTRIBUTION
        
        if rolling == False and self.fit_data == None:
            print('Error: No static model has been fit!')
            estimated_VaR = pd.DataFrame()
        elif rolling == True and self.rolling_fit_data == None:
            estimated_VaR = pd.DataFrame()
            print('Error: No rolling model has been fit!')

        elif rolling == False and self.fit_data != None:
            sigma = self.fit_data['sigma']
            table = 'robust_coefs' if use_robust == True else 'coefs'
            shape = self.fit_data[table].loc[self.fit_data[table].index == 'shape'].values[0][0] 
            mu = self.fit_data[table].loc[self.fit_data[table].index == 'mu'].values[0][0]

            quantile = robjects.conversion.rpy2py(r(f"qdist(distribution = 'std' , shape = {shape} , p = {alpha})"))[0]
            estimated_VaR = mu + sigma*quantile
            estimated_VaR.columns = [f'VaR ({alpha})']

        elif rolling == True and self.rolling_fit_data != None:

            fit_data = self.rolling_fit_data['coefs'].copy()
            fit_data['quantiles'] = np.array([robjects.conversion.rpy2py(r(f"qdist(distribution = 'std' , shape = {si} , p = {alpha})"))[0] for si in fit_data.Shape.values])

            estimated_VaR = fit_data.Mu + fit_data.Sigma * fit_data.quantiles
            estimated_VaR = pd.DataFrame(estimated_VaR.values, columns=[f'VaR ({alpha})'], index=estimated_VaR.index)

        self.historical_var_estimate = estimated_VaR

        return estimated_VaR

    def evaluateVaR(self, rolling=False, model_alpha=0.05, binomial_alpha=0.05, use_robust=True):
        
        est_var = self.historicalVaR(rolling=rolling, alpha=model_alpha, use_robust=use_robust)
        if rolling == True:
            diff = est_var[est_var.columns[0]].iloc[self.roll_start_idx:].values - self.dataset[self.endog].iloc[self.roll_start_idx:].values #np.sum(est_var.iloc[self.roll_start_idx:] > self.dataset[self.endog].iloc[self.roll_start_idx:])
        else:
            diff = est_var[est_var.columns[0]].values - self.dataset[self.endog].values

        fail_rate = ((diff > 0).sum()/len(diff))

        lower, upper = proportion_confint(int(len(diff)*model_alpha), len(diff), alpha=binomial_alpha, method='normal')# binom_confint(len(diff), model_alpha, alpha=binomial_alpha)

        did_pass = True if fail_rate > lower and fail_rate < upper else False
        
        if rolling == True:
            self.fixed_var_performance_data = {'fail_rate':fail_rate, 'good_model': did_pass , 'lower_ci':lower, 'upper_ci':upper}
        elif rolling == False:
            self.rolling_var_performance_data = {'fail_rate':fail_rate, 'good_model': did_pass , 'lower_ci':lower, 'upper_ci':upper}
        else:
            pass

        return {'fail_rate':fail_rate, 'good_model': did_pass , 'lower_ci':lower, 'upper_ci':upper}

    def fitOLS(self, data, cov_type='HC3'):

        if self.mean_exog != None and len(self.mean_exog) > 0:
            self.ols_model = sm.OLS(endog=data[self.endog],
                        exog=sm.add_constant(data[self.mean_exog], prepend=False))
            self.ols_fit = self.ols_model.fit(cov_type=cov_type)
        else:
            print('There are no exogenous mean regressors to test')
    
    def whiteTest(self):
        if self.ols_model == None or self.ols_fit == None:
            print('OLS model not fit! Returning without performing analysis')
            return {}
        
        if self.mean_exog == None:
            print('There are no exogenous mean regressors to test')
            return {}
        

        white_test = het_white(self.ols_fit.resid,  self.model.exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        white_test = dict(zip(labels, white_test))
        white_test['Heteroscedasticity'] = 'Yes' if white_test['LM-Test p-value'] <= 0.05 else 'No'

        self.white_test_results = white_test

        return white_test
    
    def adf(self, confidence_level = 0.05):
        if self.ols_model == None or self.ols_fit == None:
            print('OLS model not fit! Returning without performing analysis')
            return {}
        
        if self.mean_exog == None:
            print('There are no exogenous mean regressors to test')
            return {}
        
        adf_res = adfuller(self.ols_fit.resid)
        self.adf_sym = 'Stationary' if adf_res[1] <= confidence_level else 'Not Stationary'
        self.adf_test_results = adf_res
        return adf_res

    ### Export the fit data ###
    def exportFitData(self, tag=None, directory=None):
        if tag is None or directory is None:
            print(f'Either tag or directory is None. Cancelling Export for {self.id}')
            return
        else:
            pass

        filename = directory + self.id +'_' +tag + '.xlsx'
        with pd.ExcelWriter(filename) as writer:
            for key, df in self.fit_data.items():
                df.to_excel(writer, sheet_name=key)
    
    ### Export the fit S4 object ###
    def exportFitS4(self, tag=None, directory=None):
        if tag is None or directory is None:
            print(f'Either tag or directory is None. Cancelling Export for {self.id}')
            return
        else:
            pass

        filename = directory + self.id +'_' +tag + '.txt'
        export_string = str(self.fitS4)

        with open(filename, 'w') as file:
            file.write(export_string)
    
    ### requires language model ###
    def describeFitResults(self, prompt_text = 'Write a brief summary about the following model results which are in a tabular format. Note that trailing * asteristks indicate signifigance at 5%, 2.5%, and 1%. Also Note that numerical values enclosed in parentheses denote standard errors. Do not list the asset names in full and try to avoid discussing numerics\n'):
        data_prompt = str(self.fitS4)
        prompt = prompt_text + data_prompt
        print(f'Prompt:{prompt}')
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.6,
            max_tokens=400,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=1
        )
        #print(response)
        return response
    

########## METHODS ##########

### wrapper method to fit ARCH/GARCH models with ryp2py and the rugarch package ###
def rugarchFit(df, endog, mean_exog, variance_exog, garch_model, arma_order, garch_order, distribution_model='std'):
    global_env = robjects.globalenv

    endog_values = df[endog].values

    # logic for formatting mean model string
    if mean_exog == None:
        mean_model_str = f'list(armaOrder = c{str(arma_order)})'
    else:
        # print(mean_exog)
        # mean_exog_robject = robjects.conversion.py2rpy(df[mean_exog].values)
        # mean_regressors_matrix = robjects.r.matrix(mean_exog_robject, nrow = len(mean_exog_robject), byrow = True)
        # global_env['mean_regressors_matrix'] = mean_regressors_matrix
        # mean_model_str = f'list(armaOrder = c{str(arma_order)}, external.regressors = mean_regressors_matrix)'
    
        mean_exog_robject = robjects.conversion.py2rpy(df[mean_exog[0]].values)
        for i in range(1, len(mean_exog)):
            mean_exog_robject = robjects.r.cbind(mean_exog_robject, robjects.conversion.py2rpy(df[mean_exog[i]].values))
        mean_regressors_matrix = robjects.r.matrix(mean_exog_robject, nrow = len(mean_exog_robject), byrow = True)
        global_env['mean_regressors_matrix'] = mean_regressors_matrix
        mean_model_str = f'list(armaOrder = c{str(arma_order)}, external.regressors  = mean_regressors_matrix)'

    # logic for formatting variance model string
    if variance_exog == None:
        variance_model_str = f'list(model="{garch_model}", garchOrder = c{str(garch_order)})'
    else:
        # variance_exog_robject = robjects.conversion.py2rpy(df[variance_exog].values)
        # variance_regressors_matrix = robjects.r.matrix(variance_exog_robject, nrow = len(variance_exog_robject), byrow = True)
        # global_env['variance_regressors_matrix'] = variance_regressors_matrix
        # variance_model_str = f'list(model="{garch_model}", garchOrder = c{str(garch_order)}, external.regressors = variance_regressors_matrix)'

        variance_exog_robject = robjects.conversion.py2rpy(df[variance_exog[0]].values)
        for i in range(1, len(variance_exog)):
            variance_exog_robject = robjects.r.cbind(variance_exog_robject, robjects.conversion.py2rpy(df[variance_exog[i]].values))
        variance_regressors_matrix = robjects.r.matrix(variance_exog_robject, nrow = len(variance_exog_robject), byrow = True)
        global_env['variance_regressors_matrix'] = variance_regressors_matrix
        variance_model_str = f'list(model="{garch_model}", garchOrder = c{str(garch_order)}, external.regressors = variance_regressors_matrix)'



    garch_spec = rugarch.ugarchspec(
                        mean_model=robjects.r(mean_model_str),
                        variance_model=robjects.r(variance_model_str),
                        distribution_model=distribution_model
                        )

    fit = rugarch.ugarchfit(
            spec=garch_spec,
            data=endog_values,
            )
    
    global_env['fit'] = fit
    sigma_df = robjects.conversion.rpy2py(r('as.data.frame(sigma(fit))'))
    sigma_df.index = df.index

    residual_df = robjects.conversion.rpy2py(r('as.data.frame(residuals(fit))'))
    residual_df.index = df.index

    model_data = {
        'coefs' : robjects.conversion.rpy2py(r('as.data.frame(fit@fit[["matcoef"]])')),
        'robust_coefs' : robjects.conversion.rpy2py(r('as.data.frame(fit@fit[["robust.matcoef"]])')),
        'info_criteria' : robjects.conversion.rpy2py(r('as.data.frame(infocriteria(fit))')),
        'nyblom' : robjects.conversion.rpy2py(r('as.data.frame(nyblom(fit)$IndividualStat)')),
        'signbias' : robjects.conversion.rpy2py(r('as.data.frame(signbias(fit))')),
        'sigma' : sigma_df,
        'residuals' : residual_df
    }
    
    # This needs to be reviewed
    # ask Dr. Stewart about why R^2 is not consistent

    tss = np.sum((endog_values - np.mean(endog_values))**2)
    rss = np.sum(model_data['residuals'].values**2)
    rsquared = 1 - rss/tss #np.var(model_data['residuals'].values)/np.var(endog_values)
    log_likelihood = robjects.conversion.rpy2py(r('fit@fit$LLH'))[0]

    model_data['extra'] = pd.DataFrame(data=[rsquared,log_likelihood],index=["rsquared","logLikelihood"],columns=['V1'])

    return model_data, fit

def rugarchRollFit(df, endog, mean_exog, variance_exog, garch_model, arma_order, garch_order, distribution_model='std', nStart=100, refit=10, n_ahead=1, window_type='moving'):
    global_env = robjects.globalenv

    endog_values = df[endog].values

    # model spec formatting logic
    # logic for formatting mean model string
    if mean_exog == None:
        mean_model_str = f'list(armaOrder = c{str(arma_order)})'
    else:
        # print(mean_exog)
        # mean_exog_robject = robjects.conversion.py2rpy(df[mean_exog].values)
        # mean_regressors_matrix = robjects.r.matrix(mean_exog_robject, nrow = len(mean_exog_robject), byrow = True)
        # global_env['mean_regressors_matrix'] = mean_regressors_matrix
        # mean_model_str = f'list(armaOrder = c{str(arma_order)}, external.regressors = mean_regressors_matrix)'
    
        mean_exog_robject = robjects.conversion.py2rpy(df[mean_exog[0]].values)
        for i in range(1, len(mean_exog)):
            mean_exog_robject = robjects.r.cbind(mean_exog_robject, robjects.conversion.py2rpy(df[mean_exog[i]].values))
        mean_regressors_matrix = robjects.r.matrix(mean_exog_robject, nrow = len(mean_exog_robject), byrow = True)
        global_env['mean_regressors_matrix'] = mean_regressors_matrix
        mean_model_str = f'list(armaOrder = c{str(arma_order)}, external.regressors  = mean_regressors_matrix)'

    # logic for formatting variance model string
    if variance_exog == None:
        variance_model_str = f'list(model="{garch_model}", garchOrder = c{str(garch_order)})'
    else:
        # variance_exog_robject = robjects.conversion.py2rpy(df[variance_exog].values)
        # variance_regressors_matrix = robjects.r.matrix(variance_exog_robject, nrow = len(variance_exog_robject), byrow = True)
        # global_env['variance_regressors_matrix'] = variance_regressors_matrix
        # variance_model_str = f'list(model="{garch_model}", garchOrder = c{str(garch_order)}, external.regressors = variance_regressors_matrix)'

        variance_exog_robject = robjects.conversion.py2rpy(df[variance_exog[0]].values)
        for i in range(1, len(variance_exog)):
            variance_exog_robject = robjects.r.cbind(variance_exog_robject, robjects.conversion.py2rpy(df[variance_exog[i]].values))
        variance_regressors_matrix = robjects.r.matrix(variance_exog_robject, nrow = len(variance_exog_robject), byrow = True)
        global_env['variance_regressors_matrix'] = variance_regressors_matrix
        variance_model_str = f'list(model="{garch_model}", garchOrder = c{str(garch_order)}, external.regressors = variance_regressors_matrix)'



    garch_spec = rugarch.ugarchspec(
                        mean_model=robjects.r(mean_model_str),
                        variance_model=robjects.r(variance_model_str),
                        distribution_model=distribution_model
                        )


    rolling_fit = rugarch.ugarchroll(spec = garch_spec,
                                    data = endog_values, 
                                    n_start = nStart, 
                                    refit_every = refit,
                                    n_ahead = n_ahead,
                                    refit_window = window_type,
                                    keep_coef = True
                                    )
    
    global_env['rolling_fit'] = rolling_fit
    coef_df = robjects.conversion.rpy2py(r("as.data.frame(rolling_fit, which = 'density')")) 
    coef_df.index = df.index[nStart:]
    nan_df = pd.DataFrame([ [0]*len(coef_df.columns) for i in range(nStart)], columns = coef_df.columns, index= df.index[:nStart])
    # @dev: please review dataframe index and ensure that proper dates are used since they are currently wrong
    coef_df = pd.concat([nan_df,coef_df],axis=0)
    model_data = {
        'coefs' : coef_df,
        # 'robust_coefs' : robjects.conversion.rpy2py(r('as.data.frame(fit@fit[["robust.matcoef"]])')),
        # 'info_criteria' : robjects.conversion.rpy2py(r('as.data.frame(infocriteria(fit))')),
        # 'nyblom' : robjects.conversion.rpy2py(r('as.data.frame(nyblom(fit)$IndividualStat)')),
        # 'signbias' : robjects.conversion.rpy2py(r('as.data.frame(signbias(fit))')),
        # 'sigma' : robjects.conversion.rpy2py(r('as.data.frame(sigma(fit))')),
        # 'residuals' : robjects.conversion.rpy2py(r('as.data.frame(residuals(fit))'))
    }
    
    # # This needs to be reviewed
    # # ask Dr. Stewart about why R^2 is not consistent
    # # 

    # tss = np.sum((endog_values - np.mean(endog_values))**2)
    # rss = np.sum(model_data['residuals'].values**2)
    # rsquared = 1 - rss/tss #np.var(model_data['residuals'].values)/np.var(endog_values)
    # log_likelihood = robjects.conversion.rpy2py(r('fit@fit$LLH'))[0]

    # model_data['extra'] = pd.DataFrame(data=[rsquared,log_likelihood],index=["rsquared","logLikelihood"],columns=['V1'])

    return model_data, rolling_fit

def rugarchForecast(dataset, n_ahead ,n_roll,out_sample=0):
    #ugarchforecast(fitORspec, data = NULL, n.ahead = 10, n.roll = 0, out.sample = 0, external.forecasts = list(mregfor = NULL, vregfor = NULL)

    # rugarch.ugarchforecast(fitORspec, data = dataset, n.ahead = n_ahead, n_roll = n_roll, out_sample = out_sample, external_forecasts = list(mregfor = NULL, vregfor = NULL)
    pass


### performs a batch fit on constituents of a tradable universe relative to a benchmark symbol that is a "market" proxy ###
def batchOLSFit(benchmark, constituents, df, tag='_log_return'):
    print('Batch OLS fit')
    print('='*20)
    residuals = pd.DataFrame()
    title_str = f'| {"SYMBOl":5} | {"[slope, intercept]":27} | {"R^2":6} | Heteroscedastic residuals | stationary | AIC | BIC |'
    print('-'*len(title_str))
    print(title_str)
    print('-'*len(title_str))
    heteroscedastic_symbols = []
    for symbol in constituents:
        model = sm.OLS(endog=df[symbol+tag],
                    exog=sm.add_constant(df[benchmark+tag], prepend=False))
        fit = model.fit(cov_type='HC3')


        # heteroscedasticity test
        white_test = het_white(fit.resid,  fit.model.exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        white_test = dict(zip(labels, white_test))
        het_sym = 'Yes' if white_test['LM-Test p-value'] <= 0.05 else 'No'
        

        # stationarity test 
        adf_res = adfuller(fit.resid)
        adf_sym = 'Yes' if adf_res[1] <= 0.05 else 'No'


        if het_sym =='Yes' and adf_sym=='Yes':
            heteroscedastic_symbols.append(symbol)
        else:
            pass

        
        print(f'| {symbol:5} | {str(fit.params.values):27} | {round(fit.rsquared,3):7} | {het_sym:<24} | {adf_sym:<11} | {round(fit.aic,3):^10} | {round(fit.bic,3):^10} |')
        #print(symbol, fit.params.values, fit.rsquared)
    print('-'*len(title_str))

    return heteroscedastic_symbols

# does table formatting. Called by makeTableForPaper
def buildTableFromOutputData(target_directory, tagged_with='_GARCH_model_1.xlsx',symbols=list(), mxregs=dict(), vxregs=dict(), forLanguageModel=False):
    files = os.listdir(target_directory)
    files = [f for f in files if tagged_with in f] if tagged_with != None else files # keep the stuff we want
    files = [f for f in files if not f.startswith('~')]
    debug = pd.DataFrame(index=['slope','slope_ser','intercept','intercept_ser'])

    largest_index_symbol = None
    largest_index = 0
    iterate_symbols = symbols
    format_symbols = symbols.copy()

    unique_indicies = []

    for symbol in symbols:
        file = symbol + tagged_with
        if file in files:
            #print(target_directory+"/"+file)
            xl = pd.ExcelFile(target_directory+"/"+file)
            robust_coef = xl.parse(sheet_name='robust_coefs',index_col='Unnamed: 0')
            if robust_coef.shape[0] >= largest_index:
                largest_index_symbol = symbol
                largest_index = robust_coef.shape[0]
            else:
                pass
    

    # renaming index
    robust_coef_dict = {}
    for symbol in symbols:
        file = symbol + tagged_with
        if file in files:
            #print(target_directory+"/"+file)
            xl = pd.ExcelFile(target_directory+"/"+file)
            robust_coef = xl.parse(sheet_name='robust_coefs',index_col='Unnamed: 0')
            
            for kset in mxregs.keys():

                ksetl = kset.replace("'",'')
                ksetl = ksetl.replace("[",'')
                ksetl = ksetl.replace("]",'')
                ksetl = ksetl.replace(" ",'')

                ksetl = ksetl.split(',')

                if symbol in list(ksetl):
                    apply_mxmap = lambda string: mxregs[kset][string] if string in mxregs[kset].keys() else string
                    robust_coef.index = [apply_mxmap(d) for d in robust_coef.index]
                    #print(symbol,robust_coef.index)
                else:
                    pass
            
            for kset in vxregs.keys():

                ksetl = kset.replace("'",'')
                ksetl = ksetl.replace("[",'')
                ksetl = ksetl.replace("]",'')
                ksetl = ksetl.replace(" ",'')

                ksetl = ksetl.split(',')

                if symbol in list(ksetl):
                    apply_vxmap = lambda string: vxregs[kset][string] if string in vxregs[kset].keys() else string
                    robust_coef.index = [apply_vxmap(d) for d in robust_coef.index]
                    #print(symbol,robust_coef.index)
                else:
                    pass
        
            robust_coef_dict[symbol] = robust_coef.copy() 
    




    template = robust_coef_dict[largest_index_symbol]
    for k in robust_coef_dict.keys():
        tmp = robust_coef_dict[k].copy()
        need_to_add = [c for c in template.index if c not in tmp.index]

        if len(need_to_add) != 0:

            empty_row = dict(zip(list(template.columns), [113581321 for i in range(len(template.columns))]))
            for idx in need_to_add:
                #tmp = tmp.append(pd.Series(empty_row, index=template.columns, name=idx))
                tmp = pd.concat([tmp, pd.Series(empty_row, index=template.columns, name=idx)], axis=0)
            tmp = tmp.reindex(index = list(template.index))
            robust_coef_dict[k] = tmp.copy()

        else:
            pass



    iterate_symbols.pop(iterate_symbols.index(largest_index_symbol))
    iterate_symbols = [largest_index_symbol] + iterate_symbols

    
    for idx, symbol in enumerate(iterate_symbols):
        file = symbol + tagged_with
        if file in files:
            #print(target_directory+"/"+file)
            xl = pd.ExcelFile(target_directory+"/"+file)
            robust_coef = robust_coef_dict[symbol]#xl.parse(sheet_name='robust_coef',index_col='Unnamed: 0')

            aic = xl.parse(sheet_name='info_criteria',index_col='Unnamed: 0')
            extra = xl.parse(sheet_name='extra',index_col='Unnamed: 0')
            if idx == 0:
                table_index = list()
                for i in list(robust_coef.index): #+ list(aic.index)
                    table_index.append(i)
                    if i != 'shape':
                        table_index.append('')

                table_index.append('AIC')
                table_index.append('BIC')
                table_index.append('Rsqrd')
                table_index.append('Log Likelihood')

                table_df = pd.DataFrame(index=table_index)

            debug[symbol] = [robust_coef[' Estimate'].iloc[1], robust_coef[' Std. Error'].iloc[1],robust_coef[' Estimate'].iloc[0], robust_coef[' Std. Error'].iloc[0]]
            cval = []
            for j, vals in enumerate(robust_coef[' Estimate']):
                
                
                if robust_coef[' Estimate'].iloc[j] != 113581321:

                    if robust_coef.index[j] not in ['shape','AIC','BIC','$R^2$','Log Likelihood']:
                        pval = robust_coef['Pr(>|t|)'].iloc[j]
                        sig_str = ''
                        sig_str += '*' if pval < 0.1 else ''
                        sig_str += '*' if pval < 0.05 else ''
                        sig_str += '*' if pval < 0.01 else ''

                        if len(sig_str) > 0 and not forLanguageModel:
                            sig_str = '$^{' +sig_str + '}$'

                        cval.append(f"{round(robust_coef[' Estimate'].iloc[j],DECIMALS)}{sig_str}")
                        cval.append(f"({round(robust_coef[' Std. Error'].iloc[j],DECIMALS)})")
                    else:
                        cval.append(f"{round(robust_coef[' Estimate'].iloc[j],DECIMALS)}")
                
                else:
                    cval.append(f"--")
                    cval.append(f" ")

            cval.append(f"{round(aic['V1'].loc[aic.index == 'Akaike'].values[0],DECIMALS)}")
            cval.append(f"{round(aic['V1'].loc[aic.index == 'Bayes'].values[0],DECIMALS)}")

            cval.append(f"{round(extra['V1'].loc[extra.index == 'rsquared'].values[0],DECIMALS)}")
            cval.append(f"{round(extra['V1'].loc[extra.index == 'logLikelihood'].values[0],DECIMALS)}")


            table_df[file.replace(tagged_with,'')] = cval
    # print(symbols)
    # print(iterate_symbols)
    table_df = table_df[list(format_symbols)] # re order
    return table_df


# generate latex table
def makeTableForPaper(output_filename,target_tickers,target_directory,tagged_with,vxregs=[],table_title=''):
    df = buildTableFromOutputData(target_directory=target_directory,
                                    tagged_with=tagged_with,
                                    symbols=target_tickers,
                                    vxregs=vxregs) 
    
    latex_str = df.to_latex()

    for c in df.columns:
        latex_str = latex_str.replace(c,f'\\mc{{{c}}}')

    lstr = ''
    for i in target_tickers:
        lstr += 'l'

    latex_str = latex_str.replace('\\begin{tabular}{'+lstr+'}',f'\\resizebox{{600}}{{!}}{{\\begin{{tabular}}{{l|*{{{len(df.columns)}}}{{d{{6.6}}}}}}')
    latex_str = latex_str.replace('\\midrule','\\midrule\n\\hline')
    latex_str = latex_str.replace('mu','$\\alpha$')
    latex_str = latex_str.replace('mxreg1','$\\beta$')

    latex_str = latex_str.replace('\\\\\nomega','\\\\\n\\hline\n$\\omega$')
    latex_str = latex_str.replace('alpha1','$\\gamma$')
    latex_str = latex_str.replace('beta1','$\\delta$')

    latex_str = latex_str.replace('\\\\\nshape','$\\\\\n\\hline\nShape')

    latex_str = latex_str.replace('\\\nRsqrd','\\\n$R^2$')
    latex_str = latex_str.replace('\\\n\\bottomrule','\\\n\\hline\n\\bottomrule')


    latex_str = latex_str.replace('\{','{')
    latex_str = latex_str.replace('\}','}')
    latex_str = latex_str.replace('\\textasciicircum ','^')
    latex_str = latex_str.replace('\$','$')
    latex_str = latex_str.replace('$\\\\','\\\\')

    legend_str = 'Notes: Standard Errors shown in parentheses. p-values: *** p $\leq$ 0.01 ** p $\leq$ 0.05, * p $\leq$ 0.10'

    #latex_str.replace('\\hline\n\\bottomrule',f'\\hline\n')

    '''\\documentclass{article}
\\usepackage{lscape}
\\usepackage[utf8]{inputenc}
\\usepackage{dcolumn,booktabs}
\\newcolumntype{d}[1]{D{.}{.}{#1}}
\\newcommand\mc[1]{\multicolumn{1}{c}{#1}}
\\begin{document}
\\begin{landscape}
\\centering'''

    latex_str += '}'

    table_str = f"""\\begin{{table}}\n\\caption{{{table_title}}}"""

    table_str += latex_str
    table_str += "\\bottomrule\n \multicolumn{" +str(len(target_tickers)) +"}{c}{ \\text{Notes: Standard Errors shown in parentheses. p-values: *** p $\leq$ 0.01 ** p $\leq$ 0.05, * p $\leq$ 0.10} } \\\\\n"
    table_str += "\\end{table}\n"#\end{landscape}\n\end{document}
    table_str = table_str.replace('}\\bottomrule','')
    with open(output_filename,'w') as f:
        f.write(table_str)
    
    print('Saved Table to {}'.format(output_filename))
    #print('ALERT: Make sure to add the following LaTeX to the table file to ensure the table works')
    #print(table_str)


## unfinished work

class Section:
    def __init__(self, name:str, model_prompt:str, parameters:dict) -> None:
        self.name = name
        self.model_prompt = model_prompt
        self.parameters = parameters
        self.text = ''
    
    def populate(self,text):
        self.text = text

class Document:
    def __init__(self, name:str, sections:dict, dataset_paths:dict) -> None:
        self.name = name
        self.sections = sections
        self.sections = dataset_paths
    
    def compile(self):
        output_string = ''
        for name, section in self.sections:
            output_string += f"=== {name} ==="
            output_string += section.text

class Agent:
    def __init__(self) -> None:
        pass
    
    def isConnected(self) -> bool:
        pass