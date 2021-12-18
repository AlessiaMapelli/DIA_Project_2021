import numpy as np
from scipy import interpolate
from scipy.stats import poisson
np.random.seed(10)


class CustomerClass:
    def __init__(self, customers_parameters, fixed_parameters,arms,name=None):
        self.lambda_poisson = customers_parameters["LAMBDA_RETURNS"]
        self.n_max = customers_parameters["MAX_N_CLICKS"]
        self.slope = customers_parameters["ALPHA_N_CLICKS"]
        self.k_unif = customers_parameters["K_COST_CLICK"]
        self.probabilities = customers_parameters["CONVERSION_RATES"]
        self.fixed_cost = fixed_parameters["FIXED_COST"]
        self.arms = arms
        self.name = name

    def prob_comeback(self, x):
        return poisson.pmf(x, self.lambda_poisson)

    def n_clicks(self, bid):
        return self.n_max * (1 - np.exp(-self.slope * bid)) + np.random.normal(0, np.sqrt(2),np.size(bid))

    def cost_per_click(self, bid):
        return bid - np.random.uniform(0, bid / self.k_unif, np.size(bid))

    def conversion_rate(self, price):
        x = np.linspace(3, 15, 5)
        prob = self.probabilities + np.random.normal(0, 0.005, 1)
        cr = interpolate.interp1d(x, prob,kind='quadratic')
        ret = cr(price)
        if (ret>1):
            ret=1
        if (ret<0):
            ret=0
        return ret

    def margin(self,price):
        return self.conversion_rate(price)*(price-self.fixed_cost)*(1+np.random.poisson(self.lambda_poisson))

    def conversion_rate_mean(self, price):
        x = np.linspace(3, 15, 5)
        prob = self.probabilities
        cr = interpolate.interp1d(x, prob, kind='quadratic')
        ret = cr(price)
        if (ret > 1):
            ret = 1
        if (ret < 0):
            ret = 0
        return ret

    def cost_per_click_mean(self, bid):
        return bid - (0.5*bid/self.k_unif)

    def n_clicks_mean(self,bid):
        return self.n_max * (1 - np.exp(- self.slope * bid))

    def margin_mean(self,price):
        return self.conversion_rate_mean(price) * (price - self.fixed_cost) * (1 + self.lambda_poisson)

#OBJECTIVE FUNCTION
def obj_fun(c, p, b):
    obj = c.n_clicks(b)*(c.margin(p)-c.cost_per_click(b))
    return obj

def obj_fun_mean(c,p,b):
    obj = c.n_clicks_mean(b) * (c.margin_mean(p) - c.cost_per_click_mean(b))
    return obj