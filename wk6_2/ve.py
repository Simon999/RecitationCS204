import time
import numpy as np

def probability(variable):
    return 0.6 if variable else 0.4

# Enumeration approach
def enumeration_approach(variables):
    total_prob = 0
    for state in range(2**len(variables)):
        prob = 1
        for i, variable in enumerate(variables):
            bit = (state >> i) & 1
            prob *= probability(bit)
        total_prob += prob
    return total_prob

# Variable elimination approach
def variable_elimination_approach(variables):
    if not variables:
        return 1
    # Eliminate one variable at a time
    prob_true = probability(True) * variable_elimination_approach(variables[1:])
    prob_false = probability(False) * variable_elimination_approach(variables[1:])
    return prob_true + prob_false

# Number of variables
num_variables = 20

variables = [True] * num_variables  # Example variables

# Time enumeration approach
start_time = time.time()
enumeration_result = enumeration_approach(variables)
enumeration_time = time.time() - start_time

# Time variable elimination approach
start_time = time.time()
variable_elimination_result = variable_elimination_approach(variables)
variable_elimination_time = time.time() - start_time

print(f"Enumeration approach result: {enumeration_result}, Time: {enumeration_time} seconds")
print(f"Variable elimination approach result: {variable_elimination_result}, Time: {variable_elimination_time} seconds")
