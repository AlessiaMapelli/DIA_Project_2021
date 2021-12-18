import numpy as np

from Customer_Definition import *
from Customer_Class import *
import matplotlib.pyplot as plt

np.random.seed(10)

colors = ['r','g','b']

# plot of the interpolated conversion rate
plt.figure(figsize=(8, 4))
x = np.linspace(3, 15, 1000)
for i in range(len(customer_classes)):
    y = np.zeros(len(x))
    for j in range(len(x)):
        y[j] = customer_classes[i].conversion_rate(x[j])
    plt.plot(x, y , color=colors[i])
plt.title("Conversion Rates")
plt.xlabel("Price (€)")
plt.ylabel("Conversion Rate")

plt.legend(['Not Student - Over 60', 'Student - Under 60', 'Not Student - Under 60'])
plt.show()


# plot of number of clicks
plt.figure(figsize=(8, 4))
bid = np.linspace(0, 3, 1000)
for i in range(len(customer_classes)):
    plt.plot(bid, customer_classes[i].n_clicks(bid), color=colors[i])
plt.title("Number of clicks")
plt.xlabel("Bid (€)")
plt.ylabel("Number of clicks")

plt.legend(['Not Student - Over 60', 'Student - Under 60', 'Not Student - Under 60'])
plt.show()


# plot of cost per click
plt.figure(figsize=(8, 4))
bid = np.linspace(0, 3, 1000)
for i in range(len(customer_classes)):
    plt.plot(bid, customer_classes[i].cost_per_click(bid), color=colors[i])
plt.plot(bid, bid,'k')
plt.title("Cost per click")
plt.xlabel("Bid (€)")
plt.ylabel("Cost per click")

plt.legend(['Not Student - Over 60', 'Student - Under 60', 'Not Student - Under 60', 'bid'])
plt.show()


# plot of the distribution probability of number of come back
x = np.linspace(0, 30, 31)
plt.figure(figsize=(8, 4))
for i in range(len(customer_classes)):
    plt.scatter(x, customer_classes[i].prob_comeback(x), color=colors[i])
plt.title("Probability of return in the next 30 days")
plt.xlabel("Days")
plt.ylabel("Probability of return")

plt.legend(['Not Student - Over 60', 'Student - Under 60', 'Not Student - Under 60'])
plt.show()

plt.figure()
ax = plt.axes(projection='3d')
for c in customer_classes:
    z = np.array(np.zeros((np.size(prices),np.size(bids))))
    for k in range(len(bids)):
        for j in range(len(prices)):
                z[j, k] = obj_fun(c,prices[j],bids[k])
    i = customer_classes.index(c)
    ax.plot_wireframe(prices, bids, z, color=colors[i])
ax.set_xlabel('price')
ax.set_ylabel('bid')
ax.set_zlabel('obj function')
ax.set_title("Obj function")

plt.legend(['Not Student - Over 60', 'Student - Under 60', 'Not Student - Under 60'])
plt.show()

