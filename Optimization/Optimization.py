import numpy as np
from Optimization.Optimizer import *

np.random.seed(10)

p = prices
b = bids
optimal_values = {
    'over 60' : Optimizer(p=p,b=b,costumers=[customer_classes[0]]),
    'student' : Optimizer(p=p,b=b,costumers=[customer_classes[1]]),
    'under60_nonstudent': Optimizer(p=p,b=b,costumers=[customer_classes[2]]),
    'aggregated': Optimizer(p=p,b=b,costumers=customer_classes)
}

print(optimal_values)
