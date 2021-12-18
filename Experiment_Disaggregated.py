from matplotlib import pyplot as plt

from Optimization.Optimization import *
from Price_Bid_Joint.JointEnvironment import JointEnvironment
from Price_Bid_Joint.GTSLearnerJoint import GTSLearner
from Price_Bid_Joint.TSLearnerJoint import *

n_bids = len(bids)
n_prices = len(prices)
T = 365
n_experiments = 50

joint_rewards_per_experiment = []
opt_value = 0
optimal = optimal_values['over 60']
print('Optimal Bid over 60: ', optimal[1])
print('Optimal Price over 60: ', optimal[0])
opt_value += optimal[4]
optimal = optimal_values['student']
print('Optimal Bid student: ', optimal[1])
print('Optimal Price student: ', optimal[0])
opt_value += optimal[4]
optimal = optimal_values['under60_nonstudent']
print('Optimal Bid under 60: ', optimal[1])
print('Optimal Price under 60: ', optimal[0])
opt_value += optimal[4]
print('Optimal solution objective function value: ', opt_value)


for e in range(n_experiments):
    disaggregated_joint_rewards_per_experiment = np.zeros(T)
    print("### Experiment {}".format(e), ' ###')
    for i in range(len(customer_classes)):
        env = JointEnvironment(bids=bids, prices=prices, considered_classes=[customer_classes[i]])
        gts_learner = GTSLearner(n_arms=n_bids)
        bird = 0
        ts_learner = TSLearner(n_prices, bird)
        for t in range(T):
            # GTS - BID
            if t == 0:
                pulled_arm_bid = gts_learner.pull_arm(0, 0, 0)
            else:
                pulled_arm_bid = gts_learner.pull_arm(price_reward_vector[0]/bid_reward_vector[2], pulled_arm_price, price_reward_vector[2])
            bid_reward_vector = env.round_bid(pulled_arm_bid)
            # reward_vector = [rew_ndc, rew_cpc, total_clicks, aggregated_cpc]

            # TS - PRICE
            pulled_arm_price = ts_learner.pull_arm(t)

            price_reward_vector = \
                env.round_price(pulled_arm_bid, pulled_arm_price, bid_reward_vector[0], bid_reward_vector[1], t)
            # reward_vector = [ total_n_purchase, total_obj_function, total_returns]

            n_purchases = price_reward_vector[0]
            n_clicks = bid_reward_vector[2]
            cpc_aggregated = bid_reward_vector[3]
            daily_reward = price_reward_vector[1]
            n_returns = price_reward_vector[2]

            ts_learner.update(pulled_arm_price, n_purchases, n_clicks, daily_reward, n_returns, 0, t)

            bid_reward_vector.append(daily_reward)
            gts_learner.update(pulled_arm_bid, [n_clicks,cpc_aggregated,daily_reward])

        print('Pulled Bid: ', pulled_arm_bid,'  Class:', i)
        print('Pulled Price: ', pulled_arm_price, '  Class:',i)
        disaggregated_joint_rewards_per_experiment += ts_learner.collected_objective_function

    joint_rewards_per_experiment.append(disaggregated_joint_rewards_per_experiment)


# Plots
plt.figure()
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt_value - np.array(joint_rewards_per_experiment), axis=0)), color='b', linewidth=2.5)
plt.legend(["GTS + TS"])
plt.title("Cumulative Regret")
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.show()

plt.figure()
plt.xlabel("t")
plt.ylabel("Reward")
plt.plot(np.mean(np.array(joint_rewards_per_experiment), axis=0), color='b', linewidth=2.5)
plt.legend(["GTS + TS"])
plt.plot(opt_value * np.ones(len(np.mean(np.array(joint_rewards_per_experiment), axis=0))))
plt.title("Cumulative Regret")
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.show()