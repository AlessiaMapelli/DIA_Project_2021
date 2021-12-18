import numpy as np
from Bid_Optimization.Learner_Bid import *
from scipy.stats import norm
from Context.Customer_Definition import *


class GTSLearner(BiddingLearner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.means_rev = np.zeros(n_arms)
        self.sigmas_rev = np.ones(n_arms) * 50
        self.means_cpc = np.zeros(n_arms)
        self.sigma_cpc = np.ones(n_arms) * 50
        self.means_ndc = np.zeros(n_arms)
        self.sigma_ndc = np.ones(n_arms) * 50
        self.revenues_per_arm = [[] for i in range(n_arms)]
        self.ndc_per_arm = [[] for i in range(n_arms)]
        self.cpc_per_arm = [[] for i in range(n_arms)]
        self.threshold = 0.3
        self.fixed_cost = parameters["FIXED_CONSTANTS"]["FIXED_COST"]

    def eligible_arms(self):
        ret = []
        for i in range(0, self.n_arms):
            if norm.cdf(0, loc=self.means_rev[i], scale=self.sigmas_rev[i]) < self.threshold:
                ret.append(i)
        return ret

    def pull_arm(self, cr, price, returns):
        if self.t < 10:
            return self.t
        else:
            candidates = self.eligible_arms()
            price = prices[price]
            obj = []
            for i in candidates:
                cpc = np.random.normal(self.means_cpc[i], self.sigma_cpc[i])
                ndc = np.random.normal(self.means_ndc[i], self.sigma_ndc[i])
                obj.append(ndc*(cr*(price-self.fixed_cost)*(1+returns))-cpc)
            return candidates[np.argmax(obj)]

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.revenues_per_arm[pulled_arm].append(reward[2])
        self.ndc_per_arm[pulled_arm].append(reward[0])
        self.cpc_per_arm[pulled_arm].append(reward[1])
        self.means_rev[pulled_arm] = np.mean(self.revenues_per_arm[pulled_arm])
        self.means_ndc[pulled_arm] = np.mean(self.ndc_per_arm[pulled_arm])
        self.means_cpc[pulled_arm] = np.mean(self.cpc_per_arm[pulled_arm])
        n_samples = len(self.revenues_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas_rev[pulled_arm] = np.std(self.revenues_per_arm[pulled_arm]) / n_samples
            self.sigma_ndc[pulled_arm] = np.std(self.ndc_per_arm[pulled_arm]) / n_samples
            self.sigma_cpc[pulled_arm] = np.std(self.cpc_per_arm[pulled_arm]) / n_samples