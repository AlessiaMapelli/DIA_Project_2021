import numpy as np

from Context.Customer_Definition import parameters


class JointEnvironment:
    def __init__(self, bids, prices, considered_classes):
        self.bids = bids
        self.prices = prices
        self.considered_classes = considered_classes
        self.past_arms = []
        self.fixed_cost = 1
        self.past_purchase = []
        self.fixed_cost = parameters["FIXED_CONSTANTS"]["FIXED_COST"]

    def round_bid(self, pulled_arm_bid):
        played_bid = self.bids[pulled_arm_bid]
        rew_ndc = []
        rew_cpc = []
        obj = 0
        for c in self.considered_classes:
            ndc_per_class = round(c.n_clicks(played_bid)[0])
            rew_ndc.append(ndc_per_class)
            cpc_per_class = c.cost_per_click(played_bid)[0]
            rew_cpc.append(cpc_per_class)
        aggregated_cpc = 0
        for i in range(len(rew_ndc)):
            aggregated_cpc += (rew_ndc[i]/sum(rew_ndc))*rew_cpc[i]
        return [rew_ndc, rew_cpc, sum(rew_ndc), aggregated_cpc]

    def round_price(self, pulled_arm_bid, pulled_arm_price, rew_ndc, rew_cpc, t):
        self.past_arms.append(pulled_arm_price)
        played_bid = self.bids[pulled_arm_bid]
        played_price = self.prices[pulled_arm_price]
        n_purchases = []
        daily_objective_function = []
        returns = []
        for c in self.considered_classes:
            class_idx = self.considered_classes.index(c)
            cr = c.conversion_rate(played_price)
            n_purchases_per_class = np.random.binomial(rew_ndc[class_idx], cr)
            n_purchases.append(n_purchases_per_class)
            obj = n_purchases_per_class * (played_price - self.fixed_cost) - rew_ndc[class_idx] * rew_cpc[class_idx]
            if t > 29:
                returns_per_class = sum(np.random.poisson(c.lambda_poisson, size=int(self.past_purchase[t - 30][class_idx])))
                returns.append(returns_per_class)
                obj += returns_per_class * (self.prices[self.past_arms[t - 30]] - self.fixed_cost)
            daily_objective_function.append(obj)
        self.past_purchase.append(n_purchases)

        return [sum(n_purchases), sum(daily_objective_function), sum(returns)]