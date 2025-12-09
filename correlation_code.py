import numpy as np


class PearsonCoefficient:

    def __init__(self):
        self.stocknames = None
        self.dates      = []

    def read_data(self, filename : str = 'datasets/stock_prices_log.csv'):
        """
        Read our given csv file and collection group data
        
        :param self: class instance
        :param filename: csv file with log of stock prices
        :type filename: str
        """
        with open(filename, 'r') as file:
            for i, line in enumerate(file):
                seperated = line.split(sep=',')
                # First row contains stock names
                if i == 0:
                    self.stocknames = seperated[1:]
                    self.price_per_stock = {stockname : [] for stockname in self.stocknames}
                    self.dates.append(seperated[0])
                # Other rows contain the data which is appended to the dictionaries
                else:
                    self.dates.append(seperated[0])
                    for stockname, stockprice in zip(self.stocknames, seperated[1:]):
                        self.price_per_stock[stockname].append(float(stockprice))

        self.btc_prices  = self.price_per_stock['BTC-USD']
        self.gold_prices = self.price_per_stock['GLD']
    
    def calculate_averages(self):
        """
        Calculates the avarage stock prices over the whole series
        
        :param self: class instance
        """
        self.averages = {}
        # Averages by summing values and dividing by the length
        self.btc_avg  = sum(self.btc_prices)  / len(self.btc_prices)
        self.gold_avg = sum(self.gold_prices) / len(self.gold_prices)
        for stock, prices in self.price_per_stock.items():
            if stock != 'BTC-USD' and stock != 'GLD':
                self.averages[stock] = sum(self.price_per_stock[stock]) / len(self.price_per_stock[stock])


    def calculate_res_vars(self):
        """
        Calculates the residuals and variances of the time series
        
        :param self: class instance
        """
        self.gold_covs = {stock : 0 for stock in self.averages.keys()}
        self.btc_covs  = self.gold_covs.copy()

        # Does values-mean for residuals and sum((values-mean)^2) for variances
        self.stock_res =  {stock : [price - self.averages[stock] for price in self.price_per_stock[stock]] for stock in self.averages.keys()}
        self.stock_vars = {stock : sum([res**2 for res in self.stock_res[stock]]) for stock in self.averages.keys()}

        gold_res = [price - self.gold_avg for price in self.gold_prices]
        self.gold_var = sum([res**2 for res in gold_res])

        btc_res  = [price - self.btc_avg for price in self.btc_prices]
        self.btc_var  = sum([res**2 for res in btc_res])

        # Calculates the covariances between the stock prices and gold/bitcoin prices
        for stock, res in self.stock_res.items():
            for res_btc, res_gold, stock_res in zip(btc_res, gold_res, res):
                self.gold_covs[stock] += res_gold * stock_res
                self.btc_covs[stock]  += res_btc  * stock_res
    
    def pearsonr_final_step(self):
        """
        With the given variances and covariances this function calculates the pearson correlation coefficient between
        stock prices and gold/bitcoin prices
        
        :param self: class instance
        """
        self.pearson_gold = {}
        self.pearson_btc  = {}

        self.gold_scipy_r = {}

        # Divide by the product of the two variances to get the pearson correlation coefficient
        for btc_cov, gold_cov, stock, var in zip(self.btc_covs.values(), self.gold_covs.values(), self.stock_vars.keys(), self.stock_vars.values()):

            self.pearson_gold[stock] = gold_cov / (self.gold_var**0.5 * var**0.5)
            self.pearson_btc[stock]  = btc_cov / (self.btc_var**0.5 * var**0.5)

            # self.gold_scipy_r[stock] = st.pearsonr(self.gold_prices, self.price_per_stock[stock])
        
        return self.pearson_btc, self.pearson_gold
    
    def calculate_pearsonr(self, filename : str = 'datasets/stock_prices_log.csv'):
        """
        Combines our previous functions to calculate the pearson correlation coefficient of our dataset
        
        :param self: class instance
        :param filename: csv file with log of stock prices
        :type filename: str
        """
        self.read_data(filename)
        self.calculate_averages()
        self.calculate_res_vars()
        pearson_r_btc, pearson_r_gold = self.pearsonr_final_step()
        return pearson_r_btc, pearson_r_gold

    def corr_shifts(self):
        """
        Docstring for corr_shifts
        
        :param self: class instance
        """
        self.calculate_averages()

        self.corr_per_shift_gld = {stock : [] for stock in self.averages.keys()}
        self.corr_per_shift_btc = {stock : [] for stock in self.averages.keys()}
        
        self.best_corr_gold = {stock : 1 for stock in self.averages.keys()}
        self.best_corr_btc  = {stock : 1 for stock in self.averages.keys()}

        # Checks for all shifts what the best correlation coefficient gives and saves the shift which gave this
        for x in range(len(self.btc_prices[:-2])):
            shift = x
            self.btc_prices = self.btc_prices[:-1]
            self.gold_prices = self.gold_prices[:-1]

            self.btc_avg  = np.average(self.btc_prices)
            self.gold_avg = np.average(self.gold_prices)
            
            # Shifts stock prices and calculates avarage
            for stock, prices in self.price_per_stock.items():
                if stock != 'BTC-USD' and stock != 'GLD':
                    self.averages[stock] = sum(self.price_per_stock[stock][shift:]) / len(self.price_per_stock[stock][shift:])
            
            self.calculate_res_vars()
            btc_rs, gld_rs = self.pearsonr_final_step()

            # Checks whether the new correlation coefficient is higher
            for stock, btc_r, gold_r in zip(self.averages.keys(), btc_rs.values(), gld_rs.values()):
                if shift > 0:
                    if btc_r > max(self.corr_per_shift_btc[stock]):
                        self.best_corr_btc[stock] = shift + 1
                    if gold_r > max(self.corr_per_shift_gld[stock]):
                        self.best_corr_gold[stock] = shift + 1
                
                # Saves all coefficients
                self.corr_per_shift_btc[stock].append(btc_r)
                self.corr_per_shift_gld[stock].append(gold_r)
    
    def correlation_delay(self, filename : str = 'datasets/stock_prices_log.csv'):
        """
        Combines our previous functions to return the best correlation coefficient with it's corresponding shift
        
        :param self: class instance
        :param filename: csv file with log of stock prices
        :type filename: str
        """
        self.read_data(filename)
        self.corr_shifts()
        return self.corr_per_shift_btc, self.corr_per_shift_gld, self.best_corr_btc, self.best_corr_gold

    def pearson_r_and_delays(self, filename : str = 'datasets/stock_prices_log.csv'):
        """
        Final function for a representation of the results
        
        :param self: class instance
        :param filename: csv file with log of stock prices
        :type filename: str
        """
        pearson_r_gold, pearson_r_btc = self.calculate_pearsonr(filename)
        corr_shifts_btc, corr_shifts_gld, best_corr_btc, best_corr_gld = self.correlation_delay(filename)

        # Prints two dictionaries with values of pearson correlation coefficient between stock prices and gold/bitcoin prices
        print(f'The pearson correlation coefficients for bitcoin-stock correlation: {pearson_r_btc}')
        print(f'The pearson correlation coefficients for gold-stock correlation: {pearson_r_gold}')

        # Prints two lines per stock price to show best shift and it's corresponding correlation coefficient with gold/bitcoin prices
        for stock in corr_shifts_btc.keys():
            gold_delay = best_corr_gld[stock]
            btc_delay  = best_corr_btc[stock]
            print(f'Correlation delay between {stock} and gold: {gold_delay}, (r={corr_shifts_gld[stock][gold_delay-1]})')
            print(f'Correlation delay between {stock} and bitcoin: {btc_delay}, (r={corr_shifts_btc[stock][btc_delay-1]})')

pr = PearsonCoefficient()
pr.pearson_r_and_delays('datasets/stock_prices_log.csv')
