import numpy as np
from Context.Customer_Definition import *

class BiddingEnvironment():
    def __init__(self, bids, considered_classes):
        self.bids = bids
        self.considered_classes = considered_classes

    def round(self, pulled_arm_bid, pulled_arm_price):
        played_bid = bids[pulled_arm_bid]
        played_price = prices[pulled_arm_price]
        rew_ndc = []
        rew_cpc = []
        obj = 0
        for c in self.considered_classes:
            ndc_per_class = c.n_clicks(played_bid)
            rew_ndc.append(ndc_per_class)
            cpc_per_class = c.cost_per_click(played_bid)
            rew_cpc.append(cpc_per_class)
            obj += ndc_per_class * (c.margin_mean(played_price)-cpc_per_class)
        aggregated_cpc = ((rew_ndc[0] / sum(rew_ndc)) * rew_cpc[0]) + ((rew_ndc[1] / sum(rew_ndc)) * rew_cpc[1]) + (
                    (rew_ndc[2] / sum(rew_ndc)) * rew_cpc[2])

        return [sum(rew_ndc), aggregated_cpc, obj]
