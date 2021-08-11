"""

A simple set of nodes connected A->B->C->D, with one function at each, calculating
inputs/outputs

"""

import math

# A - Linear

A_slope = 1.1
A_intercept = 1.2

# B - Logistic

B_gain = 2.1
B_bias = 2.2
B_offset = 2.3

# C - Exponential

C_scale = 3.1
C_rate = 3.2
C_bias = 3.3
C_offset = 3.4

# D - Sine

D_scale = 4.0

test_values = [-1.0, 0.0, 1.0, 5.0]


def evaluate(input):
    A = A_slope * input + A_intercept
    B = 1 / (1 + math.exp(-1 * B_gain * (A + B_bias) + B_offset))
    C = C_scale * math.exp((C_rate * B) + C_bias) + C_offset
    D = D_scale * math.sin(C)
    print(f"  Input value {input}:\tA={A},\tB={B},\tC={C},\tD={D}")
    return A, B, C, D


if __name__ == "__main__":

    print("Evaluating ABCD net in Python, with values %s" % test_values)
    for i in test_values:
        evaluate(i)
