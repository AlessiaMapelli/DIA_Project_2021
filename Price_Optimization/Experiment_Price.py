import numpy as np
import matplotlib.pyplot as plt
from Environment_Price import *
from UCB_Pricing import *
from TS_Pricing import *
from Context.Customer_Definition import *
from Optimization.Optimization import *

np.random.seed(29345)

n_arms = parameters["FIXED_CONSTANTS"]['N_PRICES']
optimal = optimal_values["aggregated"]
pulled_arm_bid = optimal[1]
opt = optimal[4]
print('Optimal Price: ', optimal[0])
print('Optimal solution objective function value: ', opt)

T = 365
n_experiments = 50

ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []
gr_rewards_per_experiment = []

for experiment in range(0, n_experiments):
    print("Experiment {}".format(experiment))
    prob=np.zeros(3)
    env_ucb = Environment(n_arms, 0)
    env_ts = Environment(n_arms,0)
    ucb_learner = UCBLearner(n_arms=n_arms, bird=0)
    ts_learner = TSLearner(n_arms=n_arms, bird=0)
    for t in range(0, T):

        #UCB Learner
        pulled_arm_price_ucb = ucb_learner.pull_arm(t)
        n_purchases, n_clicks, daily_objective_function, n_returns = env_ucb.round(pulled_arm_price_ucb, pulled_arm_bid, t, customer_classes)
        ucb_learner.update(pulled_arm_price_ucb, n_purchases, n_clicks, daily_objective_function, n_returns, prob, t)

        #TS Learner
        pulled_arm_price_ts = ts_learner.pull_arm(t)
        n_purchases, n_clicks, daily_objective_function,n_returns = env_ts.round(pulled_arm_price_ts, pulled_arm_bid,t,customer_classes)
        ts_learner.update(pulled_arm_price_ts, n_purchases, n_clicks, daily_objective_function, n_returns,prob, t)

    print('Pulled Price UCB: ', pulled_arm_price_ucb)
    print('Pulled Price TS: ', pulled_arm_price_ts)

    ucb_rewards_per_experiment.append(ucb_learner.collected_objective_function)
    ts_rewards_per_experiment.append(ts_learner.collected_objective_function)

plt.figure(0)
plt.xlabel('t')
plt.ylabel('Regret')
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)), 'b')

plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.legend(['TS', 'UCB'])
plt.show()

plt.figure(0)
plt.xlabel('t')
plt.ylabel('Reward')
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')
plt.plot(opt*np.ones(len(np.mean(ucb_rewards_per_experiment, axis=0))))
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.legend(['TS', 'UCB'])
plt.show()