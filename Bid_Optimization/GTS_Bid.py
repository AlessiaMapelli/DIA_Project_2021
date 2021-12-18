import numpy as np
from Bid_Optimization.Learner_Bid import *
from scipy.stats import norm


class GTSLearner(BiddingLearner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.means_rev = np.zeros(n_arms)
        self.sigmas_rev = np.ones(n_arms) * 50
        self.revenues_per_arm = [[] for i in range(n_arms)]
        self.threshold = 0.2

    def eligible_arms(self):
        ret = []
        if self.t < 10:
            ret = range(0, self.n_arms)
        else:
            for i in range(0, self.n_arms):
                if norm.cdf(0, loc=self.means_rev[i], scale=self.sigmas_rev[i]) < self.threshold:
                    ret.append(i)
        return ret

    def pull_arm(self):
        if self.t < 10:
             return self.t
        else:
            candidates = self.eligible_arms()
            vals = []
            for i in candidates:
                vals.append(np.random.normal(self.means_rev[i], self.sigmas_rev[i]))
            pulled_arm = candidates[np.argmax(vals)]
            return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.revenues_per_arm[pulled_arm].append(reward[2])
        self.means_rev[pulled_arm] = np.mean(self.revenues_per_arm[pulled_arm])
        n_samples = len(self.revenues_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas_rev[pulled_arm] = np.std(self.revenues_per_arm[pulled_arm]) / n_samples