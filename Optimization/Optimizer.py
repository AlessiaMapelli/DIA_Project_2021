from Context.Customer_Definition import *

def Optimizer (p,b,costumers):
    values = np.array(np.zeros((len(p),len(b))))

    for i in range(len(b)):
        for j in range(len(p)):
            for c in costumers:
                values[j, i] += obj_fun_mean(c,p[j],b[i])

    best_price = min(np.where(values == values.max())[0])
    best_bid = max(np.where(values == values.max())[1])
    best_value = values.max()

    return best_price, best_bid, p[best_price],b[best_bid],best_value
