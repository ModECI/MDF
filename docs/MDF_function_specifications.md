# Specification of standard functions in ModECI v0.1
**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**
These functions are defined in https://github.com/ModECI/MDF/blob/main/src/modeci_mdf/standard_functions.py
## All functions:
 | <a href="#linear">linear</a> | <a href="#logistic">logistic</a> | <a href="#exponential">exponential</a> | <a href="#sin">sin</a> | <a href="#cos">cos</a> | <a href="#matmul">MatMul</a> | <a href="#relu">Relu</a> |
## linear
 <p><i>A linear function, calculated from a slope and an intercept</i></p>
<p><b>linear(variable0, slope, intercept)</b> = (variable0 * slope + intercept)</p>
<p>Python version: (variable0 * slope + intercept)</p>

## logistic
 <p><i>Logistic function</i></p>
<p><b>logistic(variable0, gain, bias, offset)</b> = 1/(1 + exp(-1*gain*(variable0 + bias) + offset))</p>
<p>Python version: 1/(1 + math.exp(-1*gain*(variable0 + bias) + offset))</p>

## exponential
 <p><i>Exponential function</i></p>
<p><b>exponential(variable0, scale, rate, bias, offset)</b> = scale * exp((rate * variable0) + bias) + offset</p>
<p>Python version: scale * math.exp((rate * variable0) + bias) + offset</p>

## sin
 <p><i>Sine function</i></p>
<p><b>sin(variable0, scale)</b> = scale * sin(variable0)</p>
<p>Python version: scale * math.sin(variable0)</p>

## cos
 <p><i>Cosine function</i></p>
<p><b>cos(variable0, scale)</b> = scale * cos(variable0)</p>
<p>Python version: scale * math.cos(variable0)</p>

## MatMul
 <p><i>Matrix multiplication (work in progress...)</i></p>
<p><b>MatMul(A, B)</b> = A @ B</p>
<p>Python version: A @ B</p>

## Relu
 <p><i>Rectified linear function (work in progress...)</i></p>
<p><b>Relu(A)</b> = maximum(A,0)</p>
<p>Python version: numpy.maximum(A,0)</p>
