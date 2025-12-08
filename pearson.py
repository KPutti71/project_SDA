import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as st


class PearsonCoefficient:

    def __init__(self):
        self.stocknames = None
        self.dates      = []

    def read_data(self, filename : str = 'stock_prices_log.csv'):

        with open(filename, 'r') as file:
            for i, line in enumerate(file):
                seperated = line.split(sep=',')
                if i == 0:
                    self.stocknames = seperated[1:]
                    self.price_per_stock = {stockname : [] for stockname in self.stocknames}
                    self.dates.append(seperated[0])
                else:
                    self.dates.append(seperated[0])
                    for stockname, stockprice in zip(self.stocknames, seperated[1:]):
                        self.price_per_stock[stockname].append(float(stockprice))

        self.btc_prices  = self.price_per_stock['BTC-USD']
        self.gold_prices = self.price_per_stock['GLD']
    
    def calculate_averages(self):
        self.averages = {}
        self.btc_avg  = sum(self.btc_prices)  / len(self.btc_prices)
        self.gold_avg = sum(self.gold_prices) / len(self.gold_prices)
        for stock, prices in self.price_per_stock.items():
            if stock != 'BTC-USD' and stock != 'GLD':
                self.averages[stock] = sum(self.price_per_stock[stock]) / len(self.price_per_stock[stock])


    def calculate_res_vars(self):
        self.gold_covs = {stock : 0 for stock in self.averages.keys()}
        self.btc_covs  = self.gold_covs.copy()

        self.stock_res =  {stock : [price - self.averages[stock] for price in self.price_per_stock[stock]] for stock in self.averages.keys()}
        self.stock_vars = {stock : sum([res**2 for res in self.stock_res[stock]]) for stock in self.averages.keys()}

        gold_res = [price - self.gold_avg for price in self.gold_prices]
        self.gold_var = sum([res**2 for res in gold_res])

        btc_res  = [price - self.btc_avg for price in self.btc_prices]
        self.btc_var  = sum([res**2 for res in btc_res])

        for stock, res in self.stock_res.items():
            for res_btc, res_gold, stock_res in zip(btc_res, gold_res, res):
                self.gold_covs[stock] += res_gold * stock_res
                self.btc_covs[stock]  += res_btc  * stock_res
    
    def pearsonr_final_step(self):
        self.pearson_gold = {}
        self.pearson_btc  = {}

        self.gold_scipy_r = {}

        for btc_cov, gold_cov, stock, var in zip(self.btc_covs.values(), self.gold_covs.values(), self.stock_vars.keys(), self.stock_vars.values()):

            self.pearson_gold[stock] = gold_cov / (self.gold_var**0.5 * var**0.5)
            self.pearson_btc[stock]  = btc_cov / (self.btc_var**0.5 * var**0.5)

            # self.gold_scipy_r[stock] = st.pearsonr(self.gold_prices, self.price_per_stock[stock])
        
        return self.pearson_btc, self.pearson_gold
    
    def calculate_pearsonr(self, filename : str = 'stock_prices_log.csv'):
        self.read_data(filename)
        self.calculate_averages()
        self.calculate_res_vars()
        pearson_r_btc, pearson_r_gold = self.pearsonr_final_step()
        return pearson_r_btc, pearson_r_gold

    def corr_shifts(self):
        self.calculate_averages()

        self.corr_per_shift_gld = {stock : [] for stock in self.averages.keys()}
        self.corr_per_shift_btc = {stock : [] for stock in self.averages.keys()}
        
        self.best_corr_gold = {stock : 1 for stock in self.averages.keys()}
        self.best_corr_btc  = {stock : 1 for stock in self.averages.keys()}

        for x in range(len(self.btc_prices[:-2])):
            shift = x
            self.btc_prices = self.btc_prices[:-1]
            self.gold_prices = self.gold_prices[:-1]

            self.btc_avg  = np.average(self.btc_prices)
            self.gold_avg = np.average(self.gold_prices)

            for stock, prices in self.price_per_stock.items():
                if stock != 'BTC-USD' and stock != 'GLD':
                    self.averages[stock] = sum(self.price_per_stock[stock][shift:]) / len(self.price_per_stock[stock][shift:])
            
            self.calculate_res_vars()
            btc_rs, gld_rs = self.pearsonr_final_step()

            for stock, btc_r, gold_r in zip(self.averages.keys(), btc_rs.values(), gld_rs.values()):
                if shift > 0:
                    if btc_r > max(self.corr_per_shift_btc[stock]):
                        self.best_corr_btc[stock] = shift + 1
                    if gold_r > max(self.corr_per_shift_gld[stock]):
                        self.best_corr_gold[stock] = shift + 1
                
                self.corr_per_shift_btc[stock].append(btc_r)
                self.corr_per_shift_gld[stock].append(gold_r)
    
    def correlation_delay(self, filename : str = 'stock_prices_log.csv'):
        self.read_data(filename)
        self.corr_shifts()
        return self.corr_per_shift_btc, self.corr_per_shift_gld, self.best_corr_btc, self.best_corr_gold

    def pearson_r_and_delays(self, filename : str = 'stock_prices_log.csv'):
        pearson_r_gold, pearson_r_btc = self.calculate_pearsonr(filename)
        corr_shifts_btc, corr_shifts_gld, best_corr_btc, best_corr_gld = self.correlation_delay(filename)
        print(f'The pearson correlation coefficients for bitcoin-stock correlation: {pearson_r_btc} \n The pearson correlation coefficients for gold-stock correlation: {pearson_r_gold}')
        for stock in corr_shifts_btc.keys():
            gold_delay = best_corr_gld[stock]
            btc_delay  = best_corr_btc[stock]
            print(f'correlation delay between {stock} and gold: {gold_delay}, (r={corr_shifts_gld[stock][gold_delay-1]})')
            print(f'Correlation delay between {stock} and bitcoin: {btc_delay}, (r={corr_shifts_btc[stock][btc_delay-1]})')

pr = PearsonCoefficient()
pr.pearson_r_and_delays('stock_prices_log.csv')
