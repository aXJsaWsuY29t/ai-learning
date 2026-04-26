from math import exp

# https://en.wikipedia.org/wiki/Softmax_function
def softmax(arr):
    norm_arr = [x - max(arr) for x in arr]
    exp_arr = [exp(x) for x in norm_arr] 
    sum_exp_arr = sum(exp_arr)
    return [x / sum_exp_arr for x in exp_arr]
