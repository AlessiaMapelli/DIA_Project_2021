import numpy as np
np.random.seed(29344)



class Learner:
    def __init__(self, n_arms, bird):
        self.n_arms = n_arms
        self.bird = 0
        self.t = 0
        self.purchases_per_price = x = [[] for i in range(n_arms)]
        self.clicks_per_price = x = [[] for i in range(n_arms)]
        self.collected_objective_function = []

    def update_observations(self, pulled_arm_price, pulled_arm_bid, n_purchases, n_clicks, estimated_obj_func):
        self.purchases_per_price[pulled_arm_price].append(n_purchases)
        self.clicks_per_price[pulled_arm_bid].append(n_clicks)
        self.collected_rewards = np.append(self.collected_rewards, estimated_obj_func)