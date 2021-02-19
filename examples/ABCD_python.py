'''

A simple set of nodes connected A->B->C->D, with one function at each, calculating
inputs/outputs

'''

import math

# A - Linear

A_slope = 2.1
A_intercept = 2.2

# B - Logistic

B_gain = 1.
B_bias = 0.
B_offset = 0.

# C - Exponential

C_scale = 1.
C_rate = 1.
C_bias = 0.
C_offset = 0.

# D - Sine

D_scale = 1.

test_values = [-1.,0.,1.,5.]

def evaluate(input):
    A = A_slope * input + A_intercept
    B = 1/(1+math.exp(-1*B_gain*(A + B_bias)+B_offset))
    C = C_scale * math.exp((C_rate * B) + C_bias) + C_offset
    D = D_scale * math.sin(C)
    print('  Input value %s:\tA=%s,\tB=%s,\tC=%s,\tD=%s'%(input, A, B, C, D))
    return A, B, C, D

if __name__ == '__main__':

    print('Evaluating ABCD net in Python, with values %s'%test_values)
    for i in test_values:
        evaluate(i)
