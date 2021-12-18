from Context.Customer_Definition import *
from Learner_Price import *

np.random.seed(29344)

class UCBLearner(Learner):
    def __init__(self, n_arms, bird):
        super().__init__(n_arms, bird)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf]*n_arms)
        self.fixed_cost = parameters["FIXED_CONSTANTS"]["FIXED_COST"]
        self.poisson_lambda = 0
        self.past_arms = []
        self.past_purchase = []
        self.collected_reward = []
        self.n_of_purchase = 0
        self.n_plays = np.zeros(n_arms)
        self.prob = 0
        self.value_mean = np.zeros(10)
        self.time_pulled = np.zeros(10)
        self.bird = bird

    def value(self, pulled_arm_price, estimated_conversion_rate):
        return estimated_conversion_rate * (prices[pulled_arm_price] - self.fixed_cost) * (1 + self.poisson_lambda)

    def pull_arm(self, t):
        time = t - self.bird
        if time < self.n_arms:
            pulled_arm = time
        else:
            upper_bound = self.empirical_means + self.confidence
            obj = np.zeros(self.n_arms)
            for i in range(self.n_arms):
                obj[i] = self.value(i, upper_bound[i])
            pulled_arm = np.random.choice(np.where(obj == obj.max())[0])
        self.past_arms.append(pulled_arm) #CAMBIATO: messo fuori dall'else
        return pulled_arm

    def update(self, pulled_arm_price, n_purchases, n_clicks, daily_reward, n_returns, prob, t):
        time = t - self.bird
        self.collected_reward.append(n_purchases)
        self.collected_objective_function.append(daily_reward)
        #print("daily_obj: ", daily_reward)
        if time > 29:
            self.poisson_lambda = (self.poisson_lambda * self.n_of_purchase + n_returns) / (self.n_of_purchase + self.collected_reward[time-30])
            self.n_of_purchase += self.collected_reward[time-30]

        self.empirical_means[pulled_arm_price] = (self.empirical_means[pulled_arm_price]*self.n_plays[pulled_arm_price]+n_purchases)/(self.n_plays[pulled_arm_price]+n_clicks)
        self.n_plays[pulled_arm_price] = self.n_plays[pulled_arm_price]+n_clicks
        self.prob = prob
        self.value_mean[pulled_arm_price] = (self.value_mean[pulled_arm_price]*self.time_pulled[pulled_arm_price]+daily_reward)/(self.time_pulled[pulled_arm_price]+1)
        self.time_pulled[pulled_arm_price] += 1
        # for each arm we update the confidence
        for a in range(self.n_arms):
            number_pulled = max(1, self.n_plays[a]) #non bisogna usare quello del giorno precedente?
            self.confidence[a] = (2*np.log(time+1)/number_pulled)**0.5

    def lowerbounds (self, t):
        alpha = 0.05
        time = t - self.bird
        clicks = np.sum(self.n_plays)
        low_value = self.value_mean[self.past_arms[time]] - np.sqrt(-np.log(alpha)/(2*(self.time_pulled[self.past_arms[time]]+1)))
        low_prob = 1
        if self.prob != 1:
            low_prob = self.prob - np.sqrt(-np.log(alpha)/(2*clicks))
        return low_value, low_prob