import numpy as np

from Context.Customer_Definition import *

np.random.seed(29344)


class Environment:
    def __init__(self, n_arms, bird):
        self.n_arms = n_arms
        self.fixed_cost = parameters["FIXED_CONSTANTS"]["FIXED_COST"]
        self.bird = bird
        self.past_purchases = np.zeros((3, 365))  # [user][time]
        self.past_arms = [[], [], []]  # [user][time]

    def round(self, pulled_arm_price, pulled_arm_bid, t, considered_classes):
        time = t - self.bird
        played_bid = bids[pulled_arm_bid]
        played_price = prices[pulled_arm_price]
        n_daily_clicks = []
        n_purchases = []
        daily_objective_function = []
        returns =[]
        for c in considered_classes:
            cr = c.conversion_rate(played_price)
            c_idx = customer_classes.index(c)
            n_clicks_per_class = round(c.n_clicks_mean(played_bid))
            n_daily_clicks.append(n_clicks_per_class)
            n_purchases_per_class = np.random.binomial(n_clicks_per_class, cr)
            n_purchases.append(n_purchases_per_class)
            self.past_purchases[c_idx][time] = n_purchases_per_class
            self.past_arms[c_idx].append(pulled_arm_price)
            obj = n_purchases_per_class * (played_price - self.fixed_cost) - n_clicks_per_class * c.cost_per_click_mean(played_bid)
            if time > 29:
                returns_per_class = sum(np.random.poisson(c.lambda_poisson, size=int(self.past_purchases[c_idx][time-30])))
                returns.append(returns_per_class)
                obj += returns_per_class * (prices[self.past_arms[c_idx][time-30]] - self.fixed_cost)
            daily_objective_function.append(obj)

        return sum(n_purchases), sum(n_daily_clicks), sum(daily_objective_function), sum(returns)

def check_condition(current_learners, learners, t):
    current_value = 0
    alternative_value = 0
    for learner in current_learners:
        current_value += learner.lowerbounds(t)[0]*learner.lowerbounds(t)[1]
    for learner in learners:
        alternative_value += learner.lowerbounds(t)[0]*learner.lowerbounds(t)[1]

    return current_value <= alternative_value, alternative_value, current_value