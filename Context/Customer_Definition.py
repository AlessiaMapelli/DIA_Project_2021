from Context.Customer_Class import *

prices = np.array([3, 4, 5, 7, 8, 9, 10, 12, 13, 15])
bids = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3])

parameters = {
    "FIXED_CONSTANTS": {
        "FIXED_COST": 1.5,
        "N_PRICES": 10,
        "N_BIDS": 10
    },
    "CLASS_1": {
        "MAX_N_CLICKS": 50,
        "ALPHA_N_CLICKS": 1,
        "K_COST_CLICK": 10,
        "CONVERSION_RATES": np.array([0.6,  0.5,  0.35,  0.2,  0.1]),
        "LAMBDA_RETURNS": 2
    },
    "CLASS_2": {
        "MAX_N_CLICKS": 200,
        "ALPHA_N_CLICKS": 0.5,
        "K_COST_CLICK": 6,
        "CONVERSION_RATES": np.array([0.9,  0.75,  0.4,  0.25,  0]),
        "LAMBDA_RETURNS": 5
    },
    "CLASS_3": {
        "MAX_N_CLICKS": 150,
        "ALPHA_N_CLICKS": 0.7,
        "K_COST_CLICK": 8,
        "CONVERSION_RATES": np.array([0.8,  0.7,  0.6,  0.45,  0.2]),
        "LAMBDA_RETURNS": 4
    }
}


np.random.seed(10)


notstudent_over60 = CustomerClass(parameters["CLASS_1"], parameters["FIXED_CONSTANTS"], prices,'NSM')
student_under60 = CustomerClass(parameters["CLASS_2"], parameters["FIXED_CONSTANTS"], prices,'SU')
notstudent_under60 = CustomerClass(parameters["CLASS_3"], parameters["FIXED_CONSTANTS"], prices,'NSO')

customer_classes = [notstudent_over60, student_under60, notstudent_under60]


