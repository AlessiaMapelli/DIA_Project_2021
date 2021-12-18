# fixed price wrt optimal solution
# HERE PRINT OPTIMAL

# Parameters
import numpy as np
from matplotlib import pyplot as plt

from GTS_Bid import GTSLearner
from Environment_Bid import BiddingEnvironment
from Optimization.Optimization import *

n_arms = len(bids)
T = 365
n_experiments = 50
gts_revenue_per_exp = []
optimal = optimal_values["aggregated"]
optimal_arm_bid = optimal[0]
opt_value = optimal[4]
print('Optimal Bid: ', optimal[1])
print('Optimal solution objective function value: ', opt_value)

for e in range(n_experiments):
    print("Experiment {}".format(e))
    env = BiddingEnvironment(bids=bids, considered_classes=customer_classes)
    gts_learner = GTSLearner(n_arms=n_arms)

    for t in range(T):
        # GTS
        pulled_arm = gts_learner.pull_arm()
        reward = env.round(pulled_arm,optimal_arm_bid)
        gts_learner.update(pulled_arm, reward)
    print('Pulled Bid: ', pulled_arm)
    gts_revenue_per_exp.append(gts_learner.collected_revenues)

# Plots
plt.figure()
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt_value - np.array(gts_revenue_per_exp), axis=0)), color='b', linewidth=2.5)
plt.legend(["GTS"])
plt.title("Cumulative Regret")
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.show()

plt.figure()
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(np.array(gts_revenue_per_exp), axis=0), color='b', linewidth=2.5)
plt.plot(opt_value*np.ones(len(np.mean(np.array(gts_revenue_per_exp),axis=0))))
plt.legend(["GTS"])
plt.title("Reward")
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.show()