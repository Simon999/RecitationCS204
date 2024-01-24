import numpy as np
def knapsack(values, weights, W):
    '''
You are given a set of items, each with a weight and a value, 
and a knapsack that can hold a maximum weight. 
The goal is to maximize the total value of items in your knapsack 
without exceeding the weight limit.

Given weights and values arrays, 
where weights[i] and values[i] represent 
the weight and value of the ith item, 
and an integer W representing the 
maximum weight your knapsack can carry.

Find the maximum value you can achieve without 
exceeding the weight limit of the knapsack.
    '''
    N = len(values)
    dp = np.zeros((N + 1, W + 1))

    for i in range(1, N + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                dp[i, w] = max(
                    dp[i - 1, w], 
                    dp[i - 1, w - weights[i - 1]] + values[i - 1])
            else:
                dp[i, w] = dp[i - 1, w]
        print(dp)
    return dp[N, W]


values = [60, 100, 120]
weights = [1, 2, 3]
W = 5
knapsack(values, weights, W)