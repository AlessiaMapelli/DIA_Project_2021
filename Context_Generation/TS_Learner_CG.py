from Context.Customer_Definition import *
from Price_Optimization.Learner_Price import *

np.random.seed(29344)

class TSLearner(Learner):
    def __init__(self, n_arms, bird):
        super().__init__(n_arms, bird)
        self.beta_parameteres = np.ones([n_arms, 2])
        self.fixed_cost = parameters["FIXED_CONSTANTS"]["FIXED_COST"]
        self.poisson_lambda = 0
        self.past_arms = []
        self.past_purchase = []
        self.collected_reward = []
        self.n_of_purchase = 0
        self.n_plays = np.zeros(n_arms)
        self.prob = 0
        self.empirical_means = np.zeros(10)
        self.time_pulled = np.zeros(10)
        self.bird = bird

    def value(self, pulled_arm_price, estimated_conversion_rate):
        return estimated_conversion_rate * (prices[pulled_arm_price] - self.fixed_cost) * (1 + self.poisson_lambda)

    def pull_arm(self):
        upper_bound = np.random.beta(self.beta_parameteres[:, 0], self.beta_parameteres[:, 1])
        obj = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            obj[i] = self.value(i, upper_bound[i])
        pulled_arm = np.random.choice(np.where(obj == obj.max())[0])
        self.past_arms.append(pulled_arm)
        return pulled_arm

    def update(self, pulled_arm_price, n_purchases, n_clicks, daily_reward, n_returns, prob, t):
        time = t - self.bird
        self.collected_reward.append(n_purchases)
        self.collected_objective_function.append(daily_reward)
        #print("daily_obj: ", daily_reward)
        if time > 29:
            if (self.n_of_purchase + self.collected_reward[time - 30]) > 0:
                self.poisson_lambda = (self.poisson_lambda * self.n_of_purchase + n_returns) / (self.n_of_purchase + self.collected_reward[time-30])
            self.n_of_purchase += self.collected_reward[time-30]

        self.beta_parameteres[pulled_arm_price, 0] = self.beta_parameteres[pulled_arm_price, 0] + n_purchases
        self.beta_parameteres[pulled_arm_price, 1] = self.beta_parameteres[pulled_arm_price, 1] + n_clicks - n_purchases

        self.prob = prob
        self.time_pulled[pulled_arm_price] += 1
        self.n_plays[pulled_arm_price] = self.n_plays[pulled_arm_price] + n_clicks
        # for each arm we update the confidence

    def lowerbounds (self, t):
        alpha = 0.05
        time = t - self.bird
        clicks = np.sum(self.n_plays)
        best_arm = self.past_arms[time]
        value_best_arm = np.random.beta(self.beta_parameteres[best_arm, 0], self.beta_parameteres[best_arm, 1])
        low_cr = value_best_arm - np.sqrt(
            -np.log(alpha) / (2 * self.n_plays[best_arm]))
        low_value = self.value(best_arm, low_cr)
        low_prob = 1
        if self.prob < 0.99:
            low_prob = self.prob - np.sqrt(-np.log(alpha) / (2 * clicks))
        return low_value, low_prob