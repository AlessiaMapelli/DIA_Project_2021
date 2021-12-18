import numpy as np

import matplotlib.pyplot as plt
from Environment_CG import *
from UCB_Learner_CG import *
from TS_Learner_CG import *
from Context.Customer_Definition import *
from Optimization.Optimization import *
from statistics import mode

#algorithm:
#1) for every feature:
#    2) evaluate the value after the split
#3) select the feature with the maximum value if larger that the not splitted case

n_experiments=50
fixed_cost=1
days=365

split1_mean = 0
split2_mean = 0

n_arms = parameters["FIXED_CONSTANTS"]['N_PRICES']
arms = np.linspace(3, 5, n_arms)
optimal = optimal_values["aggregated"]
bid = optimal[1]
opt_value = 0
optimal = optimal_values['over 60']
opt_value += optimal[4]
optimal = optimal_values['student']
opt_value += optimal[4]
optimal = optimal_values['under60_nonstudent']
opt_value += optimal[4]
print('Optimal Price: ', optimal[0])
print('Optimal solution objective function value: ', opt_value)

ucb_rewards_per_experiment = np.zeros((n_experiments, days))
ts_rewards_per_experiment = np.zeros((n_experiments, days))

context_period = 14

for e in range(0, n_experiments):
    print("### Experiment {}".format(e), ' ###')
    tot_clicks=0
    probs = np.zeros(3)

    arms = []

    split = 0
    current_class = [[0, 1, 2]]

    #first feature: stud/not stud
    #second feature: <60/>60

    env = [Environment(n_arms, 0)]
    ucb_learner = [[UCBLearner(n_arms, 0)], [], [], []]

    split_feature1 = 0
    split_feature2 = 0

    time_split0 = 0
    time_split1 = 0
    time_split2 = 0

    for t in range(0, days):

        clicks = np.zeros(3)
        obj_current = 0

        if split == 0:
            if time_split0 == 0:
                splitted_class_feature1 = [[1], [0, 2]]
                splitted_class_feature2 = [[0], [1, 2]]
                splitted_class_mix = [[2], [0, 1]]
                contexts = [current_class, splitted_class_feature1, splitted_class_feature2, splitted_class_mix]
            time_split0 += 1

        if split == 1:
            if time_split1 == 0:
                splitted_class_second = [[0], [1], [2]]
                contexts = [current_class, splitted_class_second]
            time_split1 += 1

        if split == 2:
            if time_split2 == 0:
                contexts = [current_class]
            time_split2 += 1

        #for i in range(len(customer_classes)):
        #    today_cliks[i] = customer_classes[i].n_clicks(bid)
        #    tot_n_click[i] += today_cliks[i]

        #mean_prob = tot_n_click/np.sum(tot_n_click)

        low_means_reward_current = []

        for context_index, context in enumerate(contexts):
            low_means_reward = []
            if (time_split0 == 1 or time_split1 == 1 or time_split2 == 1) and context_index != 0:
                env.append(Environment(n_arms, t))
            for c_index, c in enumerate(context):
                if (time_split0 == 1 or time_split1 == 1 or time_split2 == 1) and context_index != 0:
                    ucb_learner[context_index].append(UCBLearner(n_arms, t))
                    #print(context_index)
                class_clicks = 0
                class_reward = 0
                class_ret = 0
                class_obj = 0
                class_probs = 0

                pulled_arm = ucb_learner[context_index][c_index].pull_arm(t)

                for i in c:
                    reward, clicks[i], obj, ret = env[context_index].round(pulled_arm, bid, t, [customer_classes[i]])
                    #print("obj: %, user: %", obj, i)
                    class_reward += reward
                    class_clicks += clicks[i]
                    class_ret += ret
                    class_obj += obj
                    class_probs += probs[i]

                ucb_learner[context_index][c_index].update(pulled_arm, class_reward, class_clicks, class_obj, class_ret, class_probs, t)
                if context_index == 0:
                    obj_current += class_obj
        probs = (probs*tot_clicks+clicks)/(tot_clicks+sum(clicks))
        tot_clicks += sum(clicks)

        # check split condition
        if t % context_period == 0 and split != 2 and t != 0:
            value = np.zeros(len(contexts))
            for idx in range(1, len(contexts)):
                if check_condition(ucb_learner[0], ucb_learner[idx], t)[0]:
                    value[idx] = check_condition(ucb_learner[0], ucb_learner[idx], t)[1]
            max_value = value.max()
            if max_value > 0:
                split += 1
                splitted_index = np.random.choice(np.where(value == max_value)[0])
                current_class = contexts[splitted_index]
                if split == 1:
                    ucb_learner = [ucb_learner[splitted_index], []]
                else:  # if split == 2
                    ucb_learner = [ucb_learner[splitted_index]]
                env = [env[splitted_index]]

                print("Splitted context ", splitted_index, "at time ", t, "split=", split)

        ucb_rewards_per_experiment[e, t] = obj_current

    split1_mean=(split1_mean*e+time_split0)/(e+1)
    split2_mean=(split2_mean*e+time_split1)/(e+1)


mean_splitted_idx1 = []
exp_splitted1 = 0
for e in range(0, n_experiments):
    print("### Experiment {}".format(e), ' ###')
    tot_clicks=0
    probs = np.zeros(3)

    arms = []

    split = 0
    current_class = [[0, 1, 2]]

    #first feature: stud/not stud
    #second feature: <60/>60

    env = [Environment(n_arms, 0)]
    ts_learner = [[TSLearner(n_arms, 0)], [], [], []]

    split_feature1 = 0
    split_feature2 = 0

    time_split0 = 0
    time_split1 = 0
    time_split2 = 0

    for t in range(0, days):

        clicks = np.zeros(3)
        obj_current = 0

        if split == 0:
            if time_split0 == 0:
                splitted_class_feature1 = [[1], [0, 2]]
                splitted_class_feature2 = [[0], [1, 2]]
                splitted_class_mix = [[2], [0, 1]]
                contexts = [current_class, splitted_class_feature1, splitted_class_feature2, splitted_class_mix]
            time_split0 += 1

        if split == 1:
            if time_split1 == 0:
                splitted_class_second = [[0], [1], [2]]
                contexts = [current_class, splitted_class_second]
            time_split1 += 1

        if split == 2:
            if time_split2 == 0:
                contexts = [current_class]
            time_split2 += 1

        for context_index, context in enumerate(contexts):
            low_means_reward = []
            if (time_split0 == 1 or time_split1 == 1 or time_split2 == 1) and context_index != 0:
                env.append(Environment(n_arms, t))
            for c_index, c in enumerate(context):
                if (time_split0 == 1 or time_split1 == 1 or time_split2 == 1) and context_index != 0:
                    ts_learner[context_index].append(TSLearner(n_arms, t))
                    #print(context_index)
                class_clicks = 0
                class_reward = 0
                class_ret = 0
                class_obj = 0
                class_probs = 0

                pulled_arm = ts_learner[context_index][c_index].pull_arm()

                for i in c:
                    reward, clicks[i], obj, ret = env[context_index].round(pulled_arm, bid, t, [customer_classes[i]])
                    class_reward += reward
                    class_clicks += clicks[i]
                    class_ret += ret
                    class_obj += obj
                    class_probs += probs[i]

                ts_learner[context_index][c_index].update(pulled_arm, class_reward, class_clicks, class_obj, class_ret, class_probs, t)
                if context_index == 0:
                    obj_current += class_obj

        probs = (probs*tot_clicks+clicks)/(tot_clicks+sum(clicks))
        tot_clicks += sum(clicks)

        # check split condition
        if t % context_period == 0 and split != 2 and t != 0:
            value = np.zeros(len(contexts))
            for idx in range(1, len(contexts)):
                if check_condition(ts_learner[0], ts_learner[idx], t)[0]:
                    value[idx] = check_condition(ts_learner[0], ts_learner[idx], t)[1]
            max_value = value.max()
            if max_value > 0:
                split += 1
                splitted_index = np.random.choice(np.where(value == max_value)[0])
                current_class = contexts[splitted_index]
                if split == 1:
                    ts_learner = [ts_learner[splitted_index], []]
                    mean_splitted_idx1.append(splitted_index)
                    exp_splitted1 += 1
                else:  # if split == 2
                    ts_learner = [ts_learner[splitted_index]]
                env = [env[splitted_index]]

                print("Splitted context ", splitted_index, "at time ", t, "split=", split)

        ts_rewards_per_experiment[e, t] = obj_current
    split1_mean = (split1_mean * e + time_split0) / (e + 1)
    split2_mean = (split2_mean * e + time_split1) / (e + 1)

print("Mean split 1 after ", split1_mean, " days, with alternative: ", mode(mean_splitted_idx1))
print("Mean split 2 after ", split2_mean, " days")


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Reward")
#plt.plot(np.mean(ucb_rewards_per_experiment,axis=0), 'b')
plt.plot(np.mean(ts_rewards_per_experiment,axis=0), 'r')
plt.plot(np.ones(days)*opt_value)
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.legend(["TS"])
plt.show()

plt.figure(0)
plt.xlabel('t')
plt.ylabel('Regret')
#plt.plot(np.cumsum(np.mean(opt_value - ucb_rewards_per_experiment, axis=0)), 'b')
plt.plot(np.cumsum(np.mean(opt_value - ts_rewards_per_experiment, axis=0)), 'r')
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)

plt.legend(['TS'])
plt.show()