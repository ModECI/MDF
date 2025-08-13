# Specification of standard functions in ModECI v0.4
**Note: the ModECI MDF specification is still in development!** See [here](https://github.com/ModECI/MDF/issues) for ongoing discussions.
These functions are defined in Python API module <a href="https://github.com/ModECI/MDF/tree/main/src/modeci_mdf/functions">modeci_mdf.functions</a>.
## Non-ONNX Functions

- <a href="#matmul">MatMul</a>
- <a href="#relu">Relu</a>
- <a href="#arccos">arccos</a>
- <a href="#arcsin">arcsin</a>
- <a href="#arctan">arctan</a>
- <a href="#changegoal">change_goal</a>
- <a href="#checktermination">check_termination</a>
- <a href="#chunktostring">chunk_to_string</a>
- <a href="#conflictresolutionfunction">conflict_resolution_function</a>
- <a href="#cos">cos</a>
- <a href="#cosh">cosh</a>
- <a href="#driftdiffusionintegrator">drift_diffusion_integrator</a>
- <a href="#exponential">exponential</a>
- <a href="#linear">linear</a>
- <a href="#logistic">logistic</a>
- <a href="#matchproduction">match_production</a>
- <a href="#patternmatchingfunction">pattern_matching_function</a>
- <a href="#patterntostring">pattern_to_string</a>
- <a href="#retrievechunk">retrieve_chunk</a>
- <a href="#sin">sin</a>
- <a href="#sinh">sinh</a>
- <a href="#tan">tan</a>
- <a href="#tanh">tanh</a>
- <a href="#updatebuffer">update_buffer</a>
- <a href="#updategoal">update_goal</a>
- <a href="#updateretrieval">update_retrieval</a>

## ONNX Functions

- <a href="#abs">Abs</a>
- <a href="#acos">Acos</a>
- <a href="#acosh">Acosh</a>
- <a href="#add">Add</a>
- <a href="#and">And</a>
- <a href="#argmax">ArgMax</a>
- <a href="#argmin">ArgMin</a>
- <a href="#asin">Asin</a>
- <a href="#asinh">Asinh</a>
- <a href="#atan">Atan</a>
- <a href="#atanh">Atanh</a>
- <a href="#averagepool">AveragePool</a>
- <a href="#batchnormalization">BatchNormalization</a>
- <a href="#bernoulli">Bernoulli</a>
- <a href="#bitshift">BitShift</a>
- <a href="#cast">Cast</a>
- <a href="#castlike">CastLike</a>
- <a href="#ceil">Ceil</a>
- <a href="#celu">Celu</a>
- <a href="#clip">Clip</a>
- <a href="#compress">Compress</a>
- <a href="#concat">Concat</a>
- <a href="#concatfromsequence">ConcatFromSequence</a>
- <a href="#constant">Constant</a>
- <a href="#constantofshape">ConstantOfShape</a>
- <a href="#conv">Conv</a>
- <a href="#convinteger">ConvInteger</a>
- <a href="#convtranspose">ConvTranspose</a>
- <a href="#cos">Cos</a>
- <a href="#cosh">Cosh</a>
- <a href="#cumsum">CumSum</a>
- <a href="#depthtospace">DepthToSpace</a>
- <a href="#dequantizelinear">DequantizeLinear</a>
- <a href="#det">Det</a>
- <a href="#div">Div</a>
- <a href="#dropout">Dropout</a>
- <a href="#dynamicquantizelinear">DynamicQuantizeLinear</a>
- <a href="#einsum">Einsum</a>
- <a href="#elu">Elu</a>
- <a href="#equal">Equal</a>
- <a href="#erf">Erf</a>
- <a href="#exp">Exp</a>
- <a href="#expand">Expand</a>
- <a href="#eyelike">EyeLike</a>
- <a href="#flatten">Flatten</a>
- <a href="#floor">Floor</a>
- <a href="#gru">GRU</a>
- <a href="#gather">Gather</a>
- <a href="#gatherelements">GatherElements</a>
- <a href="#gathernd">GatherND</a>
- <a href="#gemm">Gemm</a>
- <a href="#globalaveragepool">GlobalAveragePool</a>
- <a href="#globallppool">GlobalLpPool</a>
- <a href="#globalmaxpool">GlobalMaxPool</a>
- <a href="#greater">Greater</a>
- <a href="#greaterorequal">GreaterOrEqual</a>
- <a href="#hardsigmoid">HardSigmoid</a>
- <a href="#hardswish">HardSwish</a>
- <a href="#hardmax">Hardmax</a>
- <a href="#identity">Identity</a>
- <a href="#instancenormalization">InstanceNormalization</a>
- <a href="#isinf">IsInf</a>
- <a href="#isnan">IsNaN</a>
- <a href="#lrn">LRN</a>
- <a href="#lstm">LSTM</a>
- <a href="#leakyrelu">LeakyRelu</a>
- <a href="#less">Less</a>
- <a href="#lessorequal">LessOrEqual</a>
- <a href="#log">Log</a>
- <a href="#logsoftmax">LogSoftmax</a>
- <a href="#lpnormalization">LpNormalization</a>
- <a href="#lppool">LpPool</a>
- <a href="#matmul">MatMul</a>
- <a href="#matmulinteger">MatMulInteger</a>
- <a href="#max">Max</a>
- <a href="#maxpool">MaxPool</a>
- <a href="#maxroipool">MaxRoiPool</a>
- <a href="#maxunpool">MaxUnpool</a>
- <a href="#mean">Mean</a>
- <a href="#meanvariancenormalization">MeanVarianceNormalization</a>
- <a href="#min">Min</a>
- <a href="#mod">Mod</a>
- <a href="#mul">Mul</a>
- <a href="#multinomial">Multinomial</a>
- <a href="#neg">Neg</a>
- <a href="#negativeloglikelihoodloss">NegativeLogLikelihoodLoss</a>
- <a href="#nonmaxsuppression">NonMaxSuppression</a>
- <a href="#nonzero">NonZero</a>
- <a href="#not">Not</a>
- <a href="#onehot">OneHot</a>
- <a href="#optional">Optional</a>
- <a href="#optionalgetelement">OptionalGetElement</a>
- <a href="#optionalhaselement">OptionalHasElement</a>
- <a href="#or">Or</a>
- <a href="#prelu">PRelu</a>
- <a href="#pad">Pad</a>
- <a href="#pow">Pow</a>
- <a href="#qlinearconv">QLinearConv</a>
- <a href="#qlinearmatmul">QLinearMatMul</a>
- <a href="#quantizelinear">QuantizeLinear</a>
- <a href="#rnn">RNN</a>
- <a href="#randomnormal">RandomNormal</a>
- <a href="#randomnormallike">RandomNormalLike</a>
- <a href="#randomuniform">RandomUniform</a>
- <a href="#randomuniformlike">RandomUniformLike</a>
- <a href="#range">Range</a>
- <a href="#reciprocal">Reciprocal</a>
- <a href="#reducel1">ReduceL1</a>
- <a href="#reducel2">ReduceL2</a>
- <a href="#reducelogsum">ReduceLogSum</a>
- <a href="#reducelogsumexp">ReduceLogSumExp</a>
- <a href="#reducemax">ReduceMax</a>
- <a href="#reducemean">ReduceMean</a>
- <a href="#reducemin">ReduceMin</a>
- <a href="#reduceprod">ReduceProd</a>
- <a href="#reducesum">ReduceSum</a>
- <a href="#reducesumsquare">ReduceSumSquare</a>
- <a href="#relu">Relu</a>
- <a href="#reshape">Reshape</a>
- <a href="#resize">Resize</a>
- <a href="#reversesequence">ReverseSequence</a>
- <a href="#roialign">RoiAlign</a>
- <a href="#round">Round</a>
- <a href="#scatter">Scatter</a>
- <a href="#scatterelements">ScatterElements</a>
- <a href="#scatternd">ScatterND</a>
- <a href="#selu">Selu</a>
- <a href="#sequenceat">SequenceAt</a>
- <a href="#sequenceconstruct">SequenceConstruct</a>
- <a href="#sequenceempty">SequenceEmpty</a>
- <a href="#sequenceerase">SequenceErase</a>
- <a href="#sequenceinsert">SequenceInsert</a>
- <a href="#sequencelength">SequenceLength</a>
- <a href="#shape">Shape</a>
- <a href="#shrink">Shrink</a>
- <a href="#sigmoid">Sigmoid</a>
- <a href="#sign">Sign</a>
- <a href="#sin">Sin</a>
- <a href="#sinh">Sinh</a>
- <a href="#size">Size</a>
- <a href="#slice">Slice</a>
- <a href="#softmax">Softmax</a>
- <a href="#softmaxcrossentropyloss">SoftmaxCrossEntropyLoss</a>
- <a href="#softplus">Softplus</a>
- <a href="#softsign">Softsign</a>
- <a href="#spacetodepth">SpaceToDepth</a>
- <a href="#split">Split</a>
- <a href="#splittosequence">SplitToSequence</a>
- <a href="#sqrt">Sqrt</a>
- <a href="#squeeze">Squeeze</a>
- <a href="#stringnormalizer">StringNormalizer</a>
- <a href="#sub">Sub</a>
- <a href="#sum">Sum</a>
- <a href="#tan">Tan</a>
- <a href="#tanh">Tanh</a>
- <a href="#tfidfvectorizer">TfIdfVectorizer</a>
- <a href="#thresholdedrelu">ThresholdedRelu</a>
- <a href="#tile">Tile</a>
- <a href="#topk">TopK</a>
- <a href="#transpose">Transpose</a>
- <a href="#trilu">Trilu</a>
- <a href="#unique">Unique</a>
- <a href="#unsqueeze">Unsqueeze</a>
- <a href="#upsample">Upsample</a>
- <a href="#where">Where</a>
- <a href="#xor">Xor</a>
<a name="matmul"></a>

## MatMul
 <p><i>Matrix multiplication (work in progress...)</i></p>
<p><b>MatMul(A, B)</b> = A @ B</p>

Python version: `A @ B`

<a name="relu"></a>

## Relu
 <p><i>Rectified linear function (work in progress...)</i></p>
<p><b>Relu(A)</b> = A * (A > 0)</p>

Python version: `A * (A > 0)`

<a name="arccos"></a>

## arccos
 <p><i>Inverse cosine function</i></p>
<p><b>arccos(variable0, scale)</b> = scale * arccos(variable0)</p>

Python version: `scale * numpy.arccos(variable0)`

<a name="arcsin"></a>

## arcsin
 <p><i>Inverse sine function</i></p>
<p><b>arcsin(variable0, scale)</b> = scale * arcsin(variable0)</p>

Python version: `scale * numpy.arcsin(variable0)`

<a name="arctan"></a>

## arctan
 <p><i>Inverse tangent function</i></p>
<p><b>arctan(variable0, scale)</b> = scale * arctan(variable0)</p>

Python version: `scale * numpy.arctan(variable0)`

<a name="changegoal"></a>

## change_goal
 <p><i>Modifies the current goal buffer using the given pattern.</i></p>
<p><b>change_goal(pattern, curr_goal)</b> = actr.change_goal(pattern,curr_goal)</p>

Python version: `actr.change_goal(pattern,curr_goal)`

<a name="checktermination"></a>

## check_termination
 <p><i>Function used to check if no production was selected.</i></p>
<p><b>check_termination(production)</b> = actr.check_termination(production)</p>

Python version: `actr.check_termination(production)`

<a name="chunktostring"></a>

## chunk_to_string
 <p><i>Converts a chunk dictionary to a string format.</i></p>
<p><b>chunk_to_string(chunk)</b> = actr.chunk_to_string(chunk)</p>

Python version: `actr.chunk_to_string(chunk)`

<a name="conflictresolutionfunction"></a>

## conflict_resolution_function
 <p><i>ACT-R conflict resolution function. Currently selects a production at random from the already matched productions, since utility values and learning
are not implemented yet.</i></p>
<p><b>conflict_resolution_function(productions)</b> = actr.conflict_resolution_function(productions)</p>

Python version: `actr.conflict_resolution_function(productions)`

<a name="cos"></a>

## cos
 <p><i>Cosine function</i></p>
<p><b>cos(variable0, scale)</b> = scale * cos(variable0)</p>

Python version: `scale * numpy.cos(variable0)`

<a name="cosh"></a>

## cosh
 <p><i>Hyperbolic cosine function</i></p>
<p><b>cosh(variable0, scale)</b> = scale * cosh(variable0)</p>

Python version: `scale * numpy.cosh(variable0)`

<a name="driftdiffusionintegrator"></a>

## drift_diffusion_integrator
 <p><i>Integrates the drift diffusion model for a single trial using and implementation of the using the Euler-Maruyama method. This is a proof of concept implementation and
is not optimized for speed.</i></p>
<p><b>drift_diffusion_integrator(starting_point, non_decision_time, drift_rate, threshold, noise, dt)</b> = ddm.drift_diffusion_integrator(starting_point,non_decision_time,drift_rate,threshold,noise,dt)</p>

Python version: `ddm.drift_diffusion_integrator(starting_point,non_decision_time,drift_rate,threshold,noise,dt)`

<a name="exponential"></a>

## exponential
 <p><i>Exponential function</i></p>
<p><b>exponential(variable0, scale, rate, bias, offset)</b> = scale * exp((rate * variable0) + bias) + offset</p>

Python version: `scale * numpy.exp((rate * variable0) + bias) + offset`

<a name="linear"></a>

## linear
 <p><i>A linear function, calculated from a slope and an intercept</i></p>
<p><b>linear(variable0, slope, intercept)</b> = (variable0 * slope + intercept)</p>

Python version: `(variable0 * slope + intercept)`

<a name="logistic"></a>

## logistic
 <p><i>Logistic function</i></p>
<p><b>logistic(variable0, gain, bias, offset)</b> = 1/(1 + exp(-1*gain*(variable0 + bias) + offset))</p>

Python version: `1/(1 + numpy.exp(-1*gain*(variable0 + bias) + offset))`

<a name="matchproduction"></a>

## match_production
 <p><i>Returns True if the production's left hand side matches the given context and adds the matching bindings to the production.</i></p>
<p><b>match_production(production, context)</b> = actr.match_production(production,context)</p>

Python version: `actr.match_production(production,context)`

<a name="abs"></a>

## Abs
 <p><i>
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where absolute value, y = abs(x), is applied to
the tensor elementwise.
</i></p>

Python version: `onnx_ops.abs(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Abs.html><i>ONNX Documentation</i></a>
<a name="acos"></a>

## Acos
 <p><i>
Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
</i></p>

Python version: `onnx_ops.acos(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Acos.html><i>ONNX Documentation</i></a>
<a name="acosh"></a>

## Acosh
 <p><i>
Calculates the hyperbolic arccosine of the given input tensor element-wise.
</i></p>

Python version: `onnx_ops.acosh(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Acosh.html><i>ONNX Documentation</i></a>
<a name="add"></a>

## Add
 <p><i>
Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
</i></p>

Python version: `onnx_ops.add(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Add.html><i>ONNX Documentation</i></a>
<a name="and"></a>

## And
 <p><i>
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.and(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__And.html><i>ONNX Documentation</i></a>
<a name="argmax"></a>

## ArgMax
 <p><i>
Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the max
is selected if the max appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.</i></p>

Python version: `onnx_ops.argmax(data, axis, keepdims, select_last_index)`

<a href=https://onnx.ai/onnx/operators/onnx__ArgMax.html><i>ONNX Documentation</i></a>
<a name="argmin"></a>

## ArgMin
 <p><i>
Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.</i></p>

Python version: `onnx_ops.argmin(data, axis, keepdims, select_last_index)`

<a href=https://onnx.ai/onnx/operators/onnx__ArgMin.html><i>ONNX Documentation</i></a>
<a name="asin"></a>

## Asin
 <p><i>
Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
</i></p>

Python version: `onnx_ops.asin(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Asin.html><i>ONNX Documentation</i></a>
<a name="asinh"></a>

## Asinh
 <p><i>
Calculates the hyperbolic arcsine of the given input tensor element-wise.
</i></p>

Python version: `onnx_ops.asinh(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Asinh.html><i>ONNX Documentation</i></a>
<a name="atan"></a>

## Atan
 <p><i>
Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
</i></p>

Python version: `onnx_ops.atan(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Atan.html><i>ONNX Documentation</i></a>
<a name="atanh"></a>

## Atanh
 <p><i>
Calculates the hyperbolic arctangent of the given input tensor element-wise.
</i></p>

Python version: `onnx_ops.atanh(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Atanh.html><i>ONNX Documentation</i></a>
<a name="averagepool"></a>

## AveragePool
 <p><i>
 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
or when ceil_mode is disabled:
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])
 ```

 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 </i></p>

Python version: `onnx_ops.averagepool(X, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides)`

<a href=https://onnx.ai/onnx/operators/onnx__AveragePool.html><i>ONNX Documentation</i></a>
<a name="batchnormalization"></a>

## BatchNormalization
 <p><i>
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

* Output case #1: Y, running_mean, running_var (training_mode=True)
* Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
```
where:
```
current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)
```
Notice that `ReduceVar` refers to the population variance, and it equals to
`sum(sqrd(x_i - x_avg)) / N`
where `N` is the population size (this formula does not use sample size `N - 1`).

The computation of ReduceMean and ReduceVar uses float to avoid overflow for float16 inputs.

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p>

Python version: `onnx_ops.batchnormalization(X, scale, B, input_mean, input_var, epsilon, momentum, training_mode)`

<a href=https://onnx.ai/onnx/operators/onnx__BatchNormalization.html><i>ONNX Documentation</i></a>
<a name="bernoulli"></a>

## Bernoulli
 <p><i>
Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).
</i></p>

Python version: `onnx_ops.bernoulli(input, dtype, seed)`

<a href=https://onnx.ai/onnx/operators/onnx__Bernoulli.html><i>ONNX Documentation</i></a>
<a name="bitshift"></a>

## BitShift
 <p><i>
Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute "direction"
is "LEFT", bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
not necessarily identical.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).</i></p>

Python version: `onnx_ops.bitshift(X, Y, direction)`

<a href=https://onnx.ai/onnx/operators/onnx__BitShift.html><i>ONNX Documentation</i></a>
<a name="cast"></a>

## Cast
 <p><i>
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
yield result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used.
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

In more detail, the conversion among numerical types should follow these rules:

* Casting from floating point to:
  * floating point: +/- infinity if OOR (out of range).
  * fixed point: undefined if OOR.
  * bool: +/- 0.0 to False; all else to True.
* Casting from fixed point to:
  * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
  * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
    signed types). For example, 200 (int16) -> -56 (int8).
  * bool: zero to False; nonzero to True.
* Casting from bool to:
  * floating point: `{1.0, 0.0}`.
  * fixed point: `{1, 0}`.
  * bool: no change.
</i></p>

Python version: `onnx_ops.cast(input, to)`

<a href=https://onnx.ai/onnx/operators/onnx__Cast.html><i>ONNX Documentation</i></a>
<a name="castlike"></a>

## CastLike
 <p><i>
The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
</i></p>

Python version: `onnx_ops.castlike(input, target_type)`

<a href=https://onnx.ai/onnx/operators/onnx__CastLike.html><i>ONNX Documentation</i></a>
<a name="ceil"></a>

## Ceil
 <p><i>
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
</i></p>

Python version: `onnx_ops.ceil(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Ceil.html><i>ONNX Documentation</i></a>
<a name="celu"></a>

## Celu
 <p><i>
Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```
</i></p>

Python version: `onnx_ops.celu(X, alpha)`

<a href=https://onnx.ai/onnx/operators/onnx__Celu.html><i>ONNX Documentation</i></a>
<a name="clip"></a>

## Clip
 <p><i>
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
</i></p>

Python version: `onnx_ops.clip(input, min, max)`

<a href=https://onnx.ai/onnx/operators/onnx__Clip.html><i>ONNX Documentation</i></a>
<a name="compress"></a>

## Compress
 <p><i>
    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    </i></p>

Python version: `onnx_ops.compress(input, condition, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__Compress.html><i>ONNX Documentation</i></a>
<a name="concat"></a>

## Concat
 <p><i>Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.</i></p>

Python version: `onnx_ops.concat(inputs, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__Concat.html><i>ONNX Documentation</i></a>
<a name="concatfromsequence"></a>

## ConcatFromSequence
 <p><i>
Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.
</i></p>

Python version: `onnx_ops.concatfromsequence(input_sequence, axis, new_axis)`

<a href=https://onnx.ai/onnx/operators/onnx__ConcatFromSequence.html><i>ONNX Documentation</i></a>
<a name="constant"></a>

## Constant
 <p><i>
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
</i></p>

Python version: `onnx_ops.constant(sparse_value, value, value_float, value_floats, value_int, value_ints, value_string, value_strings)`

<a href=https://onnx.ai/onnx/operators/onnx__Constant.html><i>ONNX Documentation</i></a>
<a name="constantofshape"></a>

## ConstantOfShape
 <p><i>
Generate a tensor with given value and shape.
</i></p>

Python version: `onnx_ops.constantofshape(input, value)`

<a href=https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html><i>ONNX Documentation</i></a>
<a name="conv"></a>

## Conv
 <p><i>
The convolution operator consumes an input tensor and a filter, and
computes the output.</i></p>

Python version: `onnx_ops.conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)`

<a href=https://onnx.ai/onnx/operators/onnx__Conv.html><i>ONNX Documentation</i></a>
<a name="convinteger"></a>

## ConvInteger
 <p><i>
The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
</i></p>

Python version: `onnx_ops.convinteger(x, w, x_zero_point, w_zero_point, auto_pad, dilations, group, kernel_shape, pads, strides)`

<a href=https://onnx.ai/onnx/operators/onnx__ConvInteger.html><i>ONNX Documentation</i></a>
<a name="convtranspose"></a>

## ConvTranspose
 <p><i>
The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    </i></p>

Python version: `onnx_ops.convtranspose(X, W, B, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides)`

<a href=https://onnx.ai/onnx/operators/onnx__ConvTranspose.html><i>ONNX Documentation</i></a>
<a name="cos"></a>

## Cos
 <p><i>
Calculates the cosine of the given input tensor, element-wise.
</i></p>

Python version: `onnx_ops.numpy.cos(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Cos.html><i>ONNX Documentation</i></a>
<a name="cosh"></a>

## Cosh
 <p><i>
Calculates the hyperbolic cosine of the given input tensor element-wise.
</i></p>

Python version: `onnx_ops.numpy.cosh(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Cosh.html><i>ONNX Documentation</i></a>
<a name="cumsum"></a>

## CumSum
 <p><i>
Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 3, 6]
exclusive=1
output = [0, 1, 3]
exclusive=0
reverse=1
output = [6, 5, 3]
exclusive=1
reverse=1
output = [5, 3, 0]
```
 </i></p>

Python version: `onnx_ops.cumsum(x, axis, exclusive, reverse)`

<a href=https://onnx.ai/onnx/operators/onnx__CumSum.html><i>ONNX Documentation</i></a>
<a name="depthtospace"></a>

## DepthToSpace
 <p><i>DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
```

In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
```
</i></p>

Python version: `onnx_ops.depthtospace(input, blocksize, mode)`

<a href=https://onnx.ai/onnx/operators/onnx__DepthToSpace.html><i>ONNX Documentation</i></a>
<a name="dequantizelinear"></a>

## DequantizeLinear
 <p><i>
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
</i></p>

Python version: `onnx_ops.dequantizelinear(x, x_scale, x_zero_point, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html><i>ONNX Documentation</i></a>
<a name="det"></a>

## Det
 <p><i>
Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
</i></p>

Python version: `onnx_ops.det(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Det.html><i>ONNX Documentation</i></a>
<a name="div"></a>

## Div
 <p><i>
Performs element-wise binary division (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
</i></p>

Python version: `onnx_ops.div(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Div.html><i>ONNX Documentation</i></a>
<a name="dropout"></a>

## Dropout
 <p><i>
Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p>

Python version: `onnx_ops.dropout(data, ratio, training_mode, seed)`

<a href=https://onnx.ai/onnx/operators/onnx__Dropout.html><i>ONNX Documentation</i></a>
<a name="dynamicquantizelinear"></a>

## DynamicQuantizeLinear
 <p><i>
A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
```

* where qmax and qmin are max and min values for quantization range i.e. [0, 255] in case of uint8
* data range is adjusted to include 0.

Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
```

* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.

Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
```

* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
</i></p>

Python version: `onnx_ops.dynamicquantizelinear(x)`

<a href=https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html><i>ONNX Documentation</i></a>
<a name="einsum"></a>

## Einsum
 <p><i>
An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation

```
output[output-term] = reduce-sum( input1[term1] * input2[term2] )
```

where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
that do not occur in the output-term.

The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
an operand tensor, and the characters within the terms correspond to operands dimensions.

This sequence may be followed by "->" to separate the left and right hand side of the equation.
If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
equation.

When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
beginning of the output. The equation string may contain space (U+0020) character.
</i></p>

Python version: `onnx_ops.einsum(Inputs, equation)`

<a href=https://onnx.ai/onnx/operators/onnx__Einsum.html><i>ONNX Documentation</i></a>
<a name="elu"></a>

## Elu
 <p><i>
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

</i></p>

Python version: `onnx_ops.elu(X, alpha)`

<a href=https://onnx.ai/onnx/operators/onnx__Elu.html><i>ONNX Documentation</i></a>
<a name="equal"></a>

## Equal
 <p><i>
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.equal(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Equal.html><i>ONNX Documentation</i></a>
<a name="erf"></a>

## Erf
 <p><i>
Computes the error function of the given input tensor element-wise.
</i></p>

Python version: `onnx_ops.erf(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Erf.html><i>ONNX Documentation</i></a>
<a name="exp"></a>

## Exp
 <p><i>
Calculates the exponential of the given input tensor, element-wise.
</i></p>

Python version: `onnx_ops.numpy.exp(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Exp.html><i>ONNX Documentation</i></a>
<a name="expand"></a>

## Expand
 <p><i>
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
</i></p>

Python version: `onnx_ops.expand(input, shape)`

<a href=https://onnx.ai/onnx/operators/onnx__Expand.html><i>ONNX Documentation</i></a>
<a name="eyelike"></a>

## EyeLike
 <p><i>
Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
</i></p>

Python version: `onnx_ops.eyelike(input, dtype, k)`

<a href=https://onnx.ai/onnx/operators/onnx__EyeLike.html><i>ONNX Documentation</i></a>
<a name="flatten"></a>

## Flatten
 <p><i>
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
</i></p>

Python version: `onnx_ops.flatten(input, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__Flatten.html><i>ONNX Documentation</i></a>
<a name="floor"></a>

## Floor
 <p><i>
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
</i></p>

Python version: `onnx_ops.floor(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Floor.html><i>ONNX Documentation</i></a>
<a name="gru"></a>

## GRU
 <p><i>
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

* `X` - input tensor
* `z` - update gate
* `r` - reset gate
* `h` - hidden gate
* `t` - time step (t-1 means previous time step)
* `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
* `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
* `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
* `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
* `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
* `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
* `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
* `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE:
  Below are optional

* Affine(x)              - alpha * x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha * Tanh(beta * x)
* HardSigmoid(x)         - min(max(alpha * x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha * (e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

* zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
* rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
* ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
* ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
* Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p>

Python version: `onnx_ops.gru(X, W, R, B, sequence_lens, initial_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout, linear_before_reset)`

<a href=https://onnx.ai/onnx/operators/onnx__GRU.html><i>ONNX Documentation</i></a>
<a name="gather"></a>

## Gather
 <p><i>
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

If `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]
indices = [
    [0, 1],
    [1, 2],
]
output = [
    [
        [1.0, 1.2],
        [2.3, 3.4],
    ],
    [
        [2.3, 3.4],
        [4.5, 5.7],
    ],
]
```

If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2, 1.9],
    [2.3, 3.4, 3.9],
    [4.5, 5.7, 5.9],
]
indices = [
    [0, 2],
]
axis = 1,
output = [
        [[1.0, 1.9]],
        [[2.3, 3.9]],
        [[4.5, 5.9]],
]
```
</i></p>

Python version: `onnx_ops.gather(data, indices, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__Gather.html><i>ONNX Documentation</i></a>
<a name="gatherelements"></a>

## GatherElements
 <p><i>

GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
data = [
    [1, 2],
    [3, 4],
]
indices = [
    [0, 0],
    [1, 0],
]
axis = 1
output = [
    [1, 1],
    [4, 3],
]
```
Example 2:
```
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
indices = [
    [1, 2, 0],
    [2, 0, 0],
]
axis = 0
output = [
    [4, 8, 3],
    [7, 2, 3],
]
```
</i></p>

Python version: `onnx_ops.gatherelements(data, indices, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__GatherElements.html><i>ONNX Documentation</i></a>
<a name="gathernd"></a>

## GatherND
 <p><i>
Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

**Example 1**

```
batch_dims = 0
data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
output  = [0,3]           # output_shape  = [2]
```

**Example 2**

```
batch_dims = 0
data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
indices = [[1],[0]]      # indices_shape = [2, 1]
output  = [[2,3],[0,1]]  # output_shape  = [2, 2]
```

**Example 3**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```

**Example 4**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]
```

**Example 5**

```
batch_dims = 1
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[1],[0]]                     # indices_shape = [2, 1]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```
</i></p>

Python version: `onnx_ops.gathernd(data, indices, batch_dims)`

<a href=https://onnx.ai/onnx/operators/onnx__GatherND.html><i>ONNX Documentation</i></a>
<a name="gemm"></a>

## Gemm
 <p><i>General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

* A' = transpose(A) if transA else A
* B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p>

Python version: `onnx_ops.gemm(A, B, C, alpha, beta, transA, transB)`

<a href=https://onnx.ai/onnx/operators/onnx__Gemm.html><i>ONNX Documentation</i></a>
<a name="globalaveragepool"></a>

## GlobalAveragePool
 <p><i>
 GlobalAveragePool consumes an input tensor X and applies average pooling across
 the values in the same channel. This is equivalent to AveragePool with kernel size
 equal to the spatial dimension of input tensor.</i></p>

Python version: `onnx_ops.globalaveragepool(X)`

<a href=https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html><i>ONNX Documentation</i></a>
<a name="globallppool"></a>

## GlobalLpPool
 <p><i>
 GlobalLpPool consumes an input tensor X and applies lp pool pooling across
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.</i></p>

Python version: `onnx_ops.globallppool(X, p)`

<a href=https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html><i>ONNX Documentation</i></a>
<a name="globalmaxpool"></a>

## GlobalMaxPool
 <p><i>
 GlobalMaxPool consumes an input tensor X and applies max pooling across
 the values in the same channel. This is equivalent to MaxPool with kernel size
 equal to the spatial dimension of input tensor.</i></p>

Python version: `onnx_ops.globalmaxpool(X)`

<a href=https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html><i>ONNX Documentation</i></a>
<a name="greater"></a>

## Greater
 <p><i>
Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.greater(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Greater.html><i>ONNX Documentation</i></a>
<a name="greaterorequal"></a>

## GreaterOrEqual
 <p><i>
Returns the tensor resulted from performing the `greater_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.greaterorequal(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html><i>ONNX Documentation</i></a>
<a name="hardsigmoid"></a>

## HardSigmoid
 <p><i>
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
</i></p>

Python version: `onnx_ops.hardsigmoid(X, alpha, beta)`

<a href=https://onnx.ai/onnx/operators/onnx__HardSigmoid.html><i>ONNX Documentation</i></a>
<a name="hardswish"></a>

## HardSwish
 <p><i>
HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.
</i></p>

Python version: `onnx_ops.hardswish(X)`

<a href=https://onnx.ai/onnx/operators/onnx__HardSwish.html><i>ONNX Documentation</i></a>
<a name="hardmax"></a>

## Hardmax
 <p><i>
The operator computes the hardmax values for the given input:

 Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

The "axis" attribute indicates the dimension along which Hardmax
will be performed. The output tensor has the same shape
and contains the Hardmax values of the corresponding input.
</i></p>

Python version: `onnx_ops.hardmax(input, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__Hardmax.html><i>ONNX Documentation</i></a>
<a name="identity"></a>

## Identity
 <p><i>Identity operator</i></p>

Python version: `onnx_ops.identity(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Identity.html><i>ONNX Documentation</i></a>
<a name="instancenormalization"></a>

## InstanceNormalization
 <p><i>
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

</i></p>

Python version: `onnx_ops.instancenormalization(input, scale, B, epsilon)`

<a href=https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html><i>ONNX Documentation</i></a>
<a name="isinf"></a>

## IsInf
 <p><i>Map infinity to true and other values to false.</i></p>

Python version: `onnx_ops.isinf(X, detect_negative, detect_positive)`

<a href=https://onnx.ai/onnx/operators/onnx__IsInf.html><i>ONNX Documentation</i></a>
<a name="isnan"></a>

## IsNaN
 <p><i>Returns which elements of the input are NaN.</i></p>

Python version: `onnx_ops.isnan(X)`

<a href=https://onnx.ai/onnx/operators/onnx__IsNaN.html><i>ONNX Documentation</i></a>
<a name="lrn"></a>

## LRN
 <p><i>
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]` in a tensor
of shape `(N x C x D1 x D2, ..., Dk)`, its region is
`{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.

`square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`,
where `max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))`.

`Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`
</i></p>

Python version: `onnx_ops.lrn(X, alpha, beta, bias, size)`

<a href=https://onnx.ai/onnx/operators/onnx__LRN.html><i>ONNX Documentation</i></a>
<a name="lstm"></a>

## LSTM
 <p><i>
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `o` - output gate
* `f` - forget gate
* `c` - cell gate
* `t` - time step (t-1 means previous time step)
* `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
* `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
* `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
* `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
* `P[iof]`  - P peephole weight vector for input, output, and forget gates
* `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
* `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
* `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
* `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
* `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

* it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
* ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
* ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
* Ct = ft (.) Ct-1 + it (.) ct
* ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
* Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p>

Python version: `onnx_ops.lstm(X, W, R, B, sequence_lens, initial_h, initial_c, P, activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget, layout)`

<a href=https://onnx.ai/onnx/operators/onnx__LSTM.html><i>ONNX Documentation</i></a>
<a name="leakyrelu"></a>

## LeakyRelu
 <p><i>
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
</i></p>

Python version: `onnx_ops.leakyrelu(X, alpha)`

<a href=https://onnx.ai/onnx/operators/onnx__LeakyRelu.html><i>ONNX Documentation</i></a>
<a name="less"></a>

## Less
 <p><i>
Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.less(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Less.html><i>ONNX Documentation</i></a>
<a name="lessorequal"></a>

## LessOrEqual
 <p><i>
Returns the tensor resulted from performing the `less_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.lessorequal(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__LessOrEqual.html><i>ONNX Documentation</i></a>
<a name="log"></a>

## Log
 <p><i>
Calculates the natural log of the given input tensor, element-wise.
</i></p>

Python version: `onnx_ops.log(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Log.html><i>ONNX Documentation</i></a>
<a name="logsoftmax"></a>

## LogSoftmax
 <p><i>
The operator computes the log of softmax values for the given input:

 LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

The "axis" attribute indicates the dimension along which LogSoftmax
will be performed. The output tensor has the same shape
and contains the LogSoftmax values of the corresponding input.
</i></p>

Python version: `onnx_ops.logsoftmax(input, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__LogSoftmax.html><i>ONNX Documentation</i></a>
<a name="lpnormalization"></a>

## LpNormalization
 <p><i>
Given a matrix, apply Lp-normalization along the provided axis.
</i></p>

Python version: `onnx_ops.lpnormalization(input, axis, p)`

<a href=https://onnx.ai/onnx/operators/onnx__LpNormalization.html><i>ONNX Documentation</i></a>
<a name="lppool"></a>

## LpPool
 <p><i>
 LpPool consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.</i></p>

Python version: `onnx_ops.lppool(X, auto_pad, kernel_shape, p, pads, strides)`

<a href=https://onnx.ai/onnx/operators/onnx__LpPool.html><i>ONNX Documentation</i></a>
<a name="matmul"></a>

## MatMul
 <p><i>
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
</i></p>

Python version: `onnx_ops.matmul(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__MatMul.html><i>ONNX Documentation</i></a>
<a name="matmulinteger"></a>

## MatMulInteger
 <p><i>
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
</i></p>

Python version: `onnx_ops.matmulinteger(A, B, a_zero_point, b_zero_point)`

<a href=https://onnx.ai/onnx/operators/onnx__MatMulInteger.html><i>ONNX Documentation</i></a>
<a name="max"></a>

## Max
 <p><i>
Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.max(data_0)`

<a href=https://onnx.ai/onnx/operators/onnx__Max.html><i>ONNX Documentation</i></a>
<a name="maxpool"></a>

## MaxPool
 <p><i>
 MaxPool consumes an input tensor X and applies max pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 max pooling consisting of computing the max on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape is calculated differently
 depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
 With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is maximum number of elements exclude pad.
 </i></p>

Python version: `onnx_ops.maxpool(X, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides)`

<a href=https://onnx.ai/onnx/operators/onnx__MaxPool.html><i>ONNX Documentation</i></a>
<a name="maxroipool"></a>

## MaxRoiPool
 <p><i>
 ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).</i></p>

Python version: `onnx_ops.maxroipool(X, rois, pooled_shape, spatial_scale)`

<a href=https://onnx.ai/onnx/operators/onnx__MaxRoiPool.html><i>ONNX Documentation</i></a>
<a name="maxunpool"></a>

## MaxUnpool
 <p><i>
MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corresponding
 pooling op that the unpooling op is trying to invert.
</i></p>

Python version: `onnx_ops.maxunpool(X, I, output_shape, kernel_shape, pads, strides)`

<a href=https://onnx.ai/onnx/operators/onnx__MaxUnpool.html><i>ONNX Documentation</i></a>
<a name="mean"></a>

## Mean
 <p><i>
Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.mean(data_0)`

<a href=https://onnx.ai/onnx/operators/onnx__Mean.html><i>ONNX Documentation</i></a>
<a name="meanvariancenormalization"></a>

## MeanVarianceNormalization
 <p><i>
      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
</i></p>

Python version: `onnx_ops.meanvariancenormalization(X, axes)`

<a href=https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html><i>ONNX Documentation</i></a>
<a name="min"></a>

## Min
 <p><i>
Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.min(data_0)`

<a href=https://onnx.ai/onnx/operators/onnx__Min.html><i>ONNX Documentation</i></a>
<a name="mod"></a>

## Mod
 <p><i>
  Performs element-wise binary modulus (with Numpy-style broadcasting support).
  The sign of the remainder is the same as that of the Divisor.

  Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
  (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
  This attribute is set to 0 by default causing the behavior to be like integer mod.
  Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

  If the input type is floating point, then `fmod` attribute must be set to 1.

  In case of dividend being zero, the results will be platform dependent.

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.mod(A, B, fmod)`

<a href=https://onnx.ai/onnx/operators/onnx__Mod.html><i>ONNX Documentation</i></a>
<a name="mul"></a>

## Mul
 <p><i>
Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
</i></p>

Python version: `onnx_ops.mul(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Mul.html><i>ONNX Documentation</i></a>
<a name="multinomial"></a>

## Multinomial
 <p><i>
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
</i></p>

Python version: `onnx_ops.multinomial(input, dtype, sample_size, seed)`

<a href=https://onnx.ai/onnx/operators/onnx__Multinomial.html><i>ONNX Documentation</i></a>
<a name="neg"></a>

## Neg
 <p><i>
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
</i></p>

Python version: `onnx_ops.neg(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Neg.html><i>ONNX Documentation</i></a>
<a name="negativeloglikelihoodloss"></a>

## NegativeLogLikelihoodLoss
 <p><i>
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
```

When an optional "weight" is provided, the sample loss is calculated as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
```

loss is zero for the case when target-value equals ignore_index.

```
loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
```

If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:

```
mean(loss), if "weight" is not provided,
```

or if weight is provided,

```
sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
```

If "reduction" attribute is set to "sum", the output is a scalar: `sum(loss)`.

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

```
// negative log likelihood loss, "none" reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
          [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]

loss = np.zeros((N, d1))
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1]

// print(loss)
// [[-3. -2.]
//  [-0. -2.]]
```

Example 2:

```
// weighted negative log likelihood loss, sum reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
        [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]
weight = [0.2, 0.3, 0.1]
loss = np.zeros((N, d1))
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1] * weight[c]

loss = np.sum(loss)
// print(loss)
// -1.1
```

Example 3:

```
// weighted negative log likelihood loss, mean reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
        [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]
weight = [0.2, 0.3, 0.1]
loss = np.zeros((N, d1))
weight_total = 0
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1] * weight[c]
        weight_total = weight_total + weight[c]

loss = np.sum(loss) / weight_total
// print(loss)
// -1.57
```
</i></p>

Python version: `onnx_ops.negativeloglikelihoodloss(input, target, weight, ignore_index, reduction)`

<a href=https://onnx.ai/onnx/operators/onnx__NegativeLogLikelihoodLoss.html><i>ONNX Documentation</i></a>
<a name="nonmaxsuppression"></a>

## NonMaxSuppression
 <p><i>
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
</i></p>

Python version: `onnx_ops.nonmaxsuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box)`

<a href=https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html><i>ONNX Documentation</i></a>
<a name="nonzero"></a>

## NonZero
 <p><i>
    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
    NonZero behaves similar to numpy.nonzero:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
    but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
</i></p>

Python version: `onnx_ops.nonzero(X)`

<a href=https://onnx.ai/onnx/operators/onnx__NonZero.html><i>ONNX Documentation</i></a>
<a name="not"></a>

## Not
 <p><i>
Returns the negation of the input tensor element-wise.
</i></p>

Python version: `onnx_ops.not(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Not.html><i>ONNX Documentation</i></a>
<a name="onehot"></a>

## OneHot
 <p><i>
    Produces a one-hot tensor based on inputs.
    The locations represented by the index values in the 'indices' input tensor will have 'on_value'
    and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
    are specified as part of required input argument 'values', which is a two-element tensor of format
    [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
    input tensor. The additional dimension is for one-hot representation. The additional dimension will
    be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
    dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
    dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
    as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
    the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
    output tensor.

    when axis = 0:
    output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

    when axis = -1:
    output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.

</i></p>

Python version: `onnx_ops.onehot(indices, depth, values, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__OneHot.html><i>ONNX Documentation</i></a>
<a name="optional"></a>

## Optional
 <p><i>
Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.
</i></p>

Python version: `onnx_ops.optional(input, type)`

<a href=https://onnx.ai/onnx/operators/onnx__Optional.html><i>ONNX Documentation</i></a>
<a name="optionalgetelement"></a>

## OptionalGetElement
 <p><i>
Outputs the element in the optional-type input. It is an error if the input value does not have an element
and the behavior is undefined in this case.
</i></p>

Python version: `onnx_ops.optionalgetelement(input)`

<a href=https://onnx.ai/onnx/operators/onnx__OptionalGetElement.html><i>ONNX Documentation</i></a>
<a name="optionalhaselement"></a>

## OptionalHasElement
 <p><i>
Returns true if the optional-type input contains an element. If it is an empty optional-type, this op returns false.
</i></p>

Python version: `onnx_ops.optionalhaselement(input)`

<a href=https://onnx.ai/onnx/operators/onnx__OptionalHasElement.html><i>ONNX Documentation</i></a>
<a name="or"></a>

## Or
 <p><i>
Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.or(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Or.html><i>ONNX Documentation</i></a>
<a name="prelu"></a>

## PRelu
 <p><i>
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).</i></p>

Python version: `onnx_ops.prelu(X, slope)`

<a href=https://onnx.ai/onnx/operators/onnx__PRelu.html><i>ONNX Documentation</i></a>
<a name="pad"></a>

## Pad
 <p><i>
Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'constant'

  constant_value = 0.0

  output =
  [
      [0.0, 0.0, 1.0, 1.2],
      [0.0, 0.0, 2.3, 3.4],
      [0.0, 0.0, 4.5, 5.7],
  ]


Example 2 (`reflect` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'reflect'

  output =
  [
      [1.0, 1.2, 1.0, 1.2],
      [2.3, 3.4, 2.3, 3.4],
      [4.5, 5.7, 4.5, 5.7],
  ]


Example 3 (`edge` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'edge'

  output =
  [
      [1.0, 1.0, 1.0, 1.2],
      [2.3, 2.3, 2.3, 3.4],
      [4.5, 4.5, 4.5, 5.7],
  ]

</i></p>

Python version: `onnx_ops.pad(data, pads, constant_value, mode)`

<a href=https://onnx.ai/onnx/operators/onnx__Pad.html><i>ONNX Documentation</i></a>
<a name="pow"></a>

## Pow
 <p><i>
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).</i></p>

Python version: `onnx_ops.pow(X, Y)`

<a href=https://onnx.ai/onnx/operators/onnx__Pow.html><i>ONNX Documentation</i></a>
<a name="qlinearconv"></a>

## QLinearConv
 <p><i>
The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
When bias is present it must be quantized using scale = input scale * weight scale and
zero point as 0.
</i></p>

Python version: `onnx_ops.qlinearconv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, auto_pad, dilations, group, kernel_shape, pads, strides)`

<a href=https://onnx.ai/onnx/operators/onnx__QLinearConv.html><i>ONNX Documentation</i></a>
<a name="qlinearmatmul"></a>

## QLinearMatMul
 <p><i>
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
</i></p>

Python version: `onnx_ops.qlinearmatmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point)`

<a href=https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html><i>ONNX Documentation</i></a>
<a name="quantizelinear"></a>

## QuantizeLinear
 <p><i>
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
</i></p>

Python version: `onnx_ops.quantizelinear(x, y_scale, y_zero_point, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html><i>ONNX Documentation</i></a>
<a name="rnn"></a>

## RNN
 <p><i>
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `t` - time step (t-1 means previous time step)
* `Wi` - W parameter weight matrix for input gate
* `Ri` - R recurrence weight matrix for input gate
* `Wbi` - W parameter bias vector for input gate
* `Rbi` - R parameter bias vector for input gate
* `WBi` - W parameter weight matrix for backward input gate
* `RBi` - R recurrence weight matrix for backward input gate
* `WBbi` - WR bias vectors for backward input gate
* `RBbi` - RR bias vectors for backward input gate
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

* Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p>

Python version: `onnx_ops.rnn(X, W, R, B, sequence_lens, initial_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, layout)`

<a href=https://onnx.ai/onnx/operators/onnx__RNN.html><i>ONNX Documentation</i></a>
<a name="randomnormal"></a>

## RandomNormal
 <p><i>
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
</i></p>

Python version: `onnx_ops.randomnormal(dtype, mean, scale, seed, shape)`

<a href=https://onnx.ai/onnx/operators/onnx__RandomNormal.html><i>ONNX Documentation</i></a>
<a name="randomnormallike"></a>

## RandomNormalLike
 <p><i>
Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
</i></p>

Python version: `onnx_ops.randomnormallike(input, dtype, mean, scale, seed)`

<a href=https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html><i>ONNX Documentation</i></a>
<a name="randomuniform"></a>

## RandomUniform
 <p><i>
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
</i></p>

Python version: `onnx_ops.randomuniform(dtype, high, low, seed, shape)`

<a href=https://onnx.ai/onnx/operators/onnx__RandomUniform.html><i>ONNX Documentation</i></a>
<a name="randomuniformlike"></a>

## RandomUniformLike
 <p><i>
Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
</i></p>

Python version: `onnx_ops.randomuniformlike(input, dtype, high, low, seed)`

<a href=https://onnx.ai/onnx/operators/onnx__RandomUniformLike.html><i>ONNX Documentation</i></a>
<a name="range"></a>

## Range
 <p><i>
Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below:

```
number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
```

The pseudocode determining the contents of the output is shown below:

```
for(int i=0; i<number_of_elements; ++i) {
  output[i] =  start + (i * delta);
}
```

Example 1

```
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]
```

Example 2

```
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]
```
</i></p>

Python version: `onnx_ops.range(start, limit, delta)`

<a href=https://onnx.ai/onnx/operators/onnx__Range.html><i>ONNX Documentation</i></a>
<a name="reciprocal"></a>

## Reciprocal
 <p><i>
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
</i></p>

Python version: `onnx_ops.reciprocal(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Reciprocal.html><i>ONNX Documentation</i></a>
<a name="reducel1"></a>

## ReduceL1
 <p><i>
Computes the L1 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducel1(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceL1.html><i>ONNX Documentation</i></a>
<a name="reducel2"></a>

## ReduceL2
 <p><i>
Computes the L2 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducel2(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceL2.html><i>ONNX Documentation</i></a>
<a name="reducelogsum"></a>

## ReduceLogSum
 <p><i>
Computes the log sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducelogsum(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html><i>ONNX Documentation</i></a>
<a name="reducelogsumexp"></a>

## ReduceLogSumExp
 <p><i>
Computes the log sum exponent of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducelogsumexp(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html><i>ONNX Documentation</i></a>
<a name="reducemax"></a>

## ReduceMax
 <p><i>
Computes the max of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducemax(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceMax.html><i>ONNX Documentation</i></a>
<a name="reducemean"></a>

## ReduceMean
 <p><i>
Computes the mean of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields undefined.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducemean(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceMean.html><i>ONNX Documentation</i></a>
<a name="reducemin"></a>

## ReduceMin
 <p><i>
Computes the min of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducemin(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceMin.html><i>ONNX Documentation</i></a>
<a name="reduceprod"></a>

## ReduceProd
 <p><i>
Computes the product of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 1.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reduceprod(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceProd.html><i>ONNX Documentation</i></a>
<a name="reducesum"></a>

## ReduceSum
 <p><i>
Computes the sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducesum(data, axes, keepdims, noop_with_empty_axes)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceSum.html><i>ONNX Documentation</i></a>
<a name="reducesumsquare"></a>

## ReduceSumSquare
 <p><i>
Computes the sum square of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.</i></p>

Python version: `onnx_ops.reducesumsquare(data, axes, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html><i>ONNX Documentation</i></a>
<a name="relu"></a>

## Relu
 <p><i>
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
</i></p>

Python version: `onnx_ops.relu(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Relu.html><i>ONNX Documentation</i></a>
<a name="reshape"></a>

## Reshape
 <p><i>
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
dimension will be set explicitly to zero (i.e. not taken from input tensor).
Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.

If the attribute 'allowzero' is set, it is invalid for the specified shape to
contain both a zero value and -1, as the value of the dimension corresponding
to -1 cannot be determined uniquely.
</i></p>

Python version: `onnx_ops.reshape(data, shape, allowzero)`

<a href=https://onnx.ai/onnx/operators/onnx__Reshape.html><i>ONNX Documentation</i></a>
<a name="resize"></a>

## Resize
 <p><i>
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.
</i></p>

Python version: `onnx_ops.resize(X, roi, scales, sizes, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode)`

<a href=https://onnx.ai/onnx/operators/onnx__Resize.html><i>ONNX Documentation</i></a>
<a name="reversesequence"></a>

## ReverseSequence
 <p><i>
Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]
</i></p>

Python version: `onnx_ops.reversesequence(input, sequence_lens, batch_axis, time_axis)`

<a href=https://onnx.ai/onnx/operators/onnx__ReverseSequence.html><i>ONNX Documentation</i></a>
<a name="roialign"></a>

## RoiAlign
 <p><i>
Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.
</i></p>

Python version: `onnx_ops.roialign(X, rois, batch_indices, mode, output_height, output_width, sampling_ratio, spatial_scale)`

<a href=https://onnx.ai/onnx/operators/onnx__RoiAlign.html><i>ONNX Documentation</i></a>
<a name="round"></a>

## Round
 <p><i>
Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halves, the rule is to round them to the nearest even integer.
If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
The output tensor has the same shape and type as the input.

Examples:
```
round([0.9]) = [1.0]
round([2.5]) = [2.0]
round([2.3]) = [2.0]
round([1.5]) = [2.0]
round([-4.5]) = [-4.0]
```
</i></p>

Python version: `onnx_ops.round(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Round.html><i>ONNX Documentation</i></a>
<a name="scatter"></a>

## Scatter
 <p><i>
This operator is deprecated. Please use ScatterElements, which provides the same functionality.

Scatter takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0,
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
</i></p>

Python version: `onnx_ops.scatter(data, indices, updates, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__Scatter.html><i>ONNX Documentation</i></a>
<a name="scatterelements"></a>

## ScatterElements
 <p><i>
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0,
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
</i></p>

Python version: `onnx_ops.scatterelements(data, indices, updates, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__ScatterElements.html><i>ONNX Documentation</i></a>
<a name="scatternd"></a>

## ScatterND
 <p><i>
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`. Note that `indices` should not have duplicate entries.
That is, two or more `updates` for the same index-location is not supported.

`indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
 `indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor. Index values are allowed to be negative, as per the usual
convention for counting backwards from the end, but are expected in the valid range.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:

    output = np.copy(data)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[indices[idx]] = updates[idx]

The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
This ensures that the output value does not depend on the iteration order.

This operator is the inverse of GatherND.

Example 1:
```
  data    = [1, 2, 3, 4, 5, 6, 7, 8]
  indices = [[4], [3], [1], [7]]
  updates = [9, 10, 11, 12]
  output  = [1, 11, 3, 10, 9, 6, 7, 12]
```

Example 2:
```
  data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
  indices = [[0], [2]]
  updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
  output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```
</i></p>

Python version: `onnx_ops.scatternd(data, indices, updates)`

<a href=https://onnx.ai/onnx/operators/onnx__ScatterND.html><i>ONNX Documentation</i></a>
<a name="selu"></a>

## Selu
 <p><i>
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
</i></p>

Python version: `onnx_ops.selu(X, alpha, gamma)`

<a href=https://onnx.ai/onnx/operators/onnx__Selu.html><i>ONNX Documentation</i></a>
<a name="sequenceat"></a>

## SequenceAt
 <p><i>
Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
</i></p>

Python version: `onnx_ops.sequenceat(input_sequence, position)`

<a href=https://onnx.ai/onnx/operators/onnx__SequenceAt.html><i>ONNX Documentation</i></a>
<a name="sequenceconstruct"></a>

## SequenceConstruct
 <p><i>
Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.
</i></p>

Python version: `onnx_ops.sequenceconstruct(inputs)`

<a href=https://onnx.ai/onnx/operators/onnx__SequenceConstruct.html><i>ONNX Documentation</i></a>
<a name="sequenceempty"></a>

## SequenceEmpty
 <p><i>
Construct an empty tensor sequence, with given data type.
</i></p>

Python version: `onnx_ops.sequenceempty(dtype)`

<a href=https://onnx.ai/onnx/operators/onnx__SequenceEmpty.html><i>ONNX Documentation</i></a>
<a name="sequenceerase"></a>

## SequenceErase
 <p><i>
Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.
</i></p>

Python version: `onnx_ops.sequenceerase(input_sequence, position)`

<a href=https://onnx.ai/onnx/operators/onnx__SequenceErase.html><i>ONNX Documentation</i></a>
<a name="sequenceinsert"></a>

## SequenceInsert
 <p><i>
Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
</i></p>

Python version: `onnx_ops.sequenceinsert(input_sequence, tensor, position)`

<a href=https://onnx.ai/onnx/operators/onnx__SequenceInsert.html><i>ONNX Documentation</i></a>
<a name="sequencelength"></a>

## SequenceLength
 <p><i>
Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
</i></p>

Python version: `onnx_ops.sequencelength(input_sequence)`

<a href=https://onnx.ai/onnx/operators/onnx__SequenceLength.html><i>ONNX Documentation</i></a>
<a name="shape"></a>

## Shape
 <p><i>
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clamped to the range [0, r-1], where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0.

Examples:

```
Input tensor with shape: [2, 3, 4]
No attributes specified.
Output: [2, 3, 4]
```

```
Input tensor with shape: [2, 3, 4]
start: -1
Output: [4]
```

```
Input tensor with shape: [2, 3, 4]
end: -1
Output: [2, 3]
```

```
Input tensor with shape: [2, 3, 4]
start: 1
end: 2
Output: [3]
```
</i></p>

Python version: `onnx_ops.shape(data, end, start)`

<a href=https://onnx.ai/onnx/operators/onnx__Shape.html><i>ONNX Documentation</i></a>
<a name="shrink"></a>

## Shrink
 <p><i>
Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.
</i></p>

Python version: `onnx_ops.shrink(input, bias, lambd)`

<a href=https://onnx.ai/onnx/operators/onnx__Shrink.html><i>ONNX Documentation</i></a>
<a name="sigmoid"></a>

## Sigmoid
 <p><i>
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
</i></p>

Python version: `onnx_ops.sigmoid(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Sigmoid.html><i>ONNX Documentation</i></a>
<a name="sign"></a>

## Sign
 <p><i>
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
</i></p>

Python version: `onnx_ops.sign(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Sign.html><i>ONNX Documentation</i></a>
<a name="sin"></a>

## Sin
 <p><i>
Calculates the sine of the given input tensor, element-wise.
</i></p>

Python version: `onnx_ops.numpy.sin(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Sin.html><i>ONNX Documentation</i></a>
<a name="sinh"></a>

## Sinh
 <p><i>
Calculates the hyperbolic sine of the given input tensor element-wise.
</i></p>

Python version: `onnx_ops.numpy.sinh(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Sinh.html><i>ONNX Documentation</i></a>
<a name="size"></a>

## Size
 <p><i>
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
</i></p>

Python version: `onnx_ops.size(data)`

<a href=https://onnx.ai/onnx/operators/onnx__Size.html><i>ONNX Documentation</i></a>
<a name="slice"></a>

## Slice
 <p><i>
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
of its input `data` tensor.

An effective `starts[i]`, `ends[i]`, and `steps[i]` must be computed for each `i`
in `[0, ... r-1]` where `r = rank(input)` as follows:

If `axes` are omitted, they are set to `[0, ..., r-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`

The effective values are initialized as `start[i] = 0`, `ends[i] = dims[i]` where
`dims` are the dimensions of `input` and `steps[i] = 1`.

All negative elements of `axes` are made non-negative by adding `r` to them, where
`r =rank(input)`.

All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
where `dims` are the dimensions of `input`. Then `start[axes[i]]` is the adjusted
`starts[i]` is clamped into the range `[0, dims[axes[i]]]` for positive stepping
and `[0, dims[axes[i]]-1]` for negative stepping.

The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
`ends[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
is clamped to `[-1, dims[axes[i]]-1]`.

Finally, `steps[axes[i]] = steps[i]`.

For slicing to the end of a dimension with unknown size, it is recommended to pass
in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.

Example 1:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
axes = [0, 1]
starts = [1, 0]
ends = [2, 3]
steps = [1, 2]
result = [
    [5, 7],
]
```

Example 2:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
starts = [0, 1]
ends = [-1, 1000]
result = [
    [2, 3, 4],
]
```
</i></p>

Python version: `onnx_ops.slice(data, starts, ends, axes, steps)`

<a href=https://onnx.ai/onnx/operators/onnx__Slice.html><i>ONNX Documentation</i></a>
<a name="softmax"></a>

## Softmax
 <p><i>
The operator computes the normalized exponential values for the given input:

 Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

The "axis" attribute indicates the dimension along which Softmax
will be performed. The output tensor has the same shape
and contains the Softmax values of the corresponding input.
</i></p>

Python version: `onnx_ops.softmax(input, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__Softmax.html><i>ONNX Documentation</i></a>
<a name="softmaxcrossentropyloss"></a>

## SoftmaxCrossEntropyLoss
 <p><i>Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

* shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.
* shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can calculated as follows:
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
```
or
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
```

loss is zero for the case when label-value equals ignore_index.
```
l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
```

where:
```
p = Softmax(scores)
y = Log(p)
c = labels[i][d1][d2]...[dk]
```

Finally, L is optionally reduced:

* If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
* If reduction = 'sum', the output is scalar: Sum(L).
* If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: `ReduceSum(L) / ReduceSum(W)`,
  where tensor W is of shape `(N, D1, D2, ..., Dk)` and `W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]`.
</i></p>

Python version: `onnx_ops.softmaxcrossentropyloss(scores, labels, weights, ignore_index, reduction)`

<a href=https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html><i>ONNX Documentation</i></a>
<a name="softplus"></a>

## Softplus
 <p><i>
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
</i></p>

Python version: `onnx_ops.softplus(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Softplus.html><i>ONNX Documentation</i></a>
<a name="softsign"></a>

## Softsign
 <p><i>
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
</i></p>

Python version: `onnx_ops.softsign(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Softsign.html><i>ONNX Documentation</i></a>
<a name="spacetodepth"></a>

## SpaceToDepth
 <p><i>SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
</i></p>

Python version: `onnx_ops.spacetodepth(input, blocksize)`

<a href=https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html><i>ONNX Documentation</i></a>
<a name="split"></a>

## Split
 <p><i>Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using input 'split'.
Otherwise, the tensor is split to equal sized parts.
</i></p>

Python version: `onnx_ops.split(input, split, axis)`

<a href=https://onnx.ai/onnx/operators/onnx__Split.html><i>ONNX Documentation</i></a>
<a name="splittosequence"></a>

## SplitToSequence
 <p><i>
Split a tensor into a sequence of tensors, along the specified 'axis'.
Lengths of the parts can be specified using the optional argument 'split'.
If the argument `split' is not specified, a default scalar value of 1
is used as the value of `split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
if possible. The last chunk alone may be smaller than 'split' if the 'input' size
along the given axis 'axis' is not divisible by 'split'.
If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
in 'split' must be equal to the dimension size of input tensor on 'axis'.
</i></p>

Python version: `onnx_ops.splittosequence(input, split, axis, keepdims)`

<a href=https://onnx.ai/onnx/operators/onnx__SplitToSequence.html><i>ONNX Documentation</i></a>
<a name="sqrt"></a>

## Sqrt
 <p><i>
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
</i></p>

Python version: `onnx_ops.sqrt(X)`

<a href=https://onnx.ai/onnx/operators/onnx__Sqrt.html><i>ONNX Documentation</i></a>
<a name="squeeze"></a>

## Squeeze
 <p><i>
Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
</i></p>

Python version: `onnx_ops.squeeze(data, axes)`

<a href=https://onnx.ai/onnx/operators/onnx__Squeeze.html><i>ONNX Documentation</i></a>
<a name="stringnormalizer"></a>

## StringNormalizer
 <p><i>
StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].
</i></p>

Python version: `onnx_ops.stringnormalizer(X, case_change_action, is_case_sensitive, locale, stopwords)`

<a href=https://onnx.ai/onnx/operators/onnx__StringNormalizer.html><i>ONNX Documentation</i></a>
<a name="sub"></a>

## Sub
 <p><i>
Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
</i></p>

Python version: `onnx_ops.sub(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Sub.html><i>ONNX Documentation</i></a>
<a name="sum"></a>

## Sum
 <p><i>
Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.sum(data_0)`

<a href=https://onnx.ai/onnx/operators/onnx__Sum.html><i>ONNX Documentation</i></a>
<a name="tan"></a>

## Tan
 <p><i>
Calculates the tangent of the given input tensor, element-wise.
</i></p>

Python version: `onnx_ops.numpy.tan(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Tan.html><i>ONNX Documentation</i></a>
<a name="tanh"></a>

## Tanh
 <p><i>
Calculates the hyperbolic tangent of the given input tensor element-wise.
</i></p>

Python version: `onnx_ops.numpy.tanh(input)`

<a href=https://onnx.ai/onnx/operators/onnx__Tanh.html><i>ONNX Documentation</i></a>
<a name="tfidfvectorizer"></a>

## TfIdfVectorizer
 <p><i>
This transform extracts n-grams from the input sequence and save them as a vector. Input can
be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.

In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
Let's consider an example. Assume that input sequence is [94, 17, 36, 12, 28] and the number of skips is 2.
The associated 2-grams are [94, 12] and [17, 28] respectively indexed by [0, 3] and [1, 4].
If the number of skips becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12, 28]
indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.

The output vector (denoted by Y) stores the count of each n-gram;
Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
between index i and the corresponding n-gram's output coordinate. If pool_int64s is [94, 17, 17, 36], ngram_indexes is [1, 0],
ngram_counts=[0, 0], then the Y[0] (first element in Y) and Y[1] (second element in Y) are the counts of [17, 36] and [94, 17],
respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
Note that we may consider all skips up to S when generating the n-grams.

The examples used above are true if mode is "TF". If mode is "IDF", all the counts larger than 1 would be truncated to 1 and
the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is "TFIDF",
this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.

Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
If pool_strings is set, the input must be a string tensor.
</i></p>

Python version: `onnx_ops.tfidfvectorizer(X, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights)`

<a href=https://onnx.ai/onnx/operators/onnx__TfIdfVectorizer.html><i>ONNX Documentation</i></a>
<a name="thresholdedrelu"></a>

## ThresholdedRelu
 <p><i>
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
</i></p>

Python version: `onnx_ops.thresholdedrelu(X, alpha)`

<a href=https://onnx.ai/onnx/operators/onnx__ThresholdedRelu.html><i>ONNX Documentation</i></a>
<a name="tile"></a>

## Tile
 <p><i>Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
</i></p>

Python version: `onnx_ops.tile(input, repeats)`

<a href=https://onnx.ai/onnx/operators/onnx__Tile.html><i>ONNX Documentation</i></a>
<a name="topk"></a>

## TopK
 <p><i>
Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:

* Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
  which contains the values of the top k elements along the specified axis
* Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
  contains the indices of the top k elements (original indices from the input
  tensor).

* If "largest" is 1 (the default value) then the k largest elements are returned.
* If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
* If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
a tiebreaker. That is, the element with the lower index will appear first.
</i></p>

Python version: `onnx_ops.topk(X, K, axis, largest, sorted)`

<a href=https://onnx.ai/onnx/operators/onnx__TopK.html><i>ONNX Documentation</i></a>
<a name="transpose"></a>

## Transpose
 <p><i>
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
</i></p>

Python version: `onnx_ops.transpose(data, perm)`

<a href=https://onnx.ai/onnx/operators/onnx__Transpose.html><i>ONNX Documentation</i></a>
<a name="trilu"></a>

## Trilu
 <p><i>
Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
The attribute "upper" determines whether the upper or lower part is retained. If set to true,
the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
Default value for the "upper" attribute is true.
Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
All other elements in the matrix are set to zero.
If k = 0, the triangular part on and above/below the main diagonal is retained.
If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
A negative k value retains the main diagonal and |k| diagonals below it.
If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
</i></p>

Python version: `onnx_ops.trilu(input, k, upper)`

<a href=https://onnx.ai/onnx/operators/onnx__Trilu.html><i>ONNX Documentation</i></a>
<a name="unique"></a>

## Unique
 <p><i>
Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
The first output tensor 'Y' contains all unique values or subtensors of the input.
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurrence in 'X'.
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'.
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
```
input_X = [2, 1, 1, 3, 4, 3]
attribute_sorted = 0
attribute_axis = None
output_Y = [2, 1, 3, 4]
output_indices = [0, 1, 3, 4]
output_inverse_indices = [0, 1, 1, 2, 3, 2]
output_counts = [1, 2, 2, 1]
```

Example 2:
```
input_X = [[1, 3], [2, 3]]
attribute_sorted = 1
attribute_axis = None
output_Y = [1, 2, 3]
output_indices = [0, 2, 1]
output_inverse_indices = [0, 2, 1, 2]
output_counts = [1, 1, 2]
```

Example 3:
```
input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
attribute_sorted = 1
attribute_axis = 0
output_Y = [[1, 0, 0], [2, 3, 4]]
output_indices = [0, 2]
output_inverse_indices = [0, 0, 1]
output_counts = [2, 1]
```

Example 4:
```
input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
            [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
attribute_sorted = 1
attribute_axis = 1
```

intermediate data are presented below for better understanding:
there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
```
A: [[1, 1], [1, 1]],
   [[0, 1], [0, 1]],
   [[2, 1], [2, 1]],
   [[0, 1], [0, 1]].
```

there are 3 unique subtensors:
```
[[1, 1], [1, 1]],
[[0, 1], [0, 1]],
[[2, 1], [2, 1]].
```

sorted unique subtensors:
```
B: [[0, 1], [0, 1]],
   [[1, 1], [1, 1]],
   [[2, 1], [2, 1]].
```

output_Y is constructed from B:
```
[[[0. 1.], [1. 1.], [2. 1.]],
 [[0. 1.], [1. 1.], [2. 1.]]]
```

output_indices is to map from B to A:
```
[1, 0, 2]
```

output_inverse_indices is to map from A to B:
```
[1, 0, 2, 0]
```

output_counts:
```
[2, 1, 1]
```
</i></p>

Python version: `onnx_ops.unique(X, axis, sorted)`

<a href=https://onnx.ai/onnx/operators/onnx__Unique.html><i>ONNX Documentation</i></a>
<a name="unsqueeze"></a>

## Unsqueeze
 <p><i>
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example, given an input tensor (`data`) of shape [3, 4, 5], then
Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.
</i></p>

Python version: `onnx_ops.unsqueeze(data, axes)`

<a href=https://onnx.ai/onnx/operators/onnx__Unsqueeze.html><i>ONNX Documentation</i></a>
<a name="upsample"></a>

## Upsample
 <p><i>
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
</i></p>

Python version: `onnx_ops.upsample(X, scales, mode)`

<a href=https://onnx.ai/onnx/operators/onnx__Upsample.html><i>ONNX Documentation</i></a>
<a name="where"></a>

## Where
 <p><i>
Return elements, either from X or Y, depending on condition.
Where behaves like
[numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).</i></p>

Python version: `onnx_ops.where(condition, X, Y)`

<a href=https://onnx.ai/onnx/operators/onnx__Where.html><i>ONNX Documentation</i></a>
<a name="xor"></a>

## Xor
 <p><i>
Returns the tensor resulted from performing the `xor` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p>

Python version: `onnx_ops.xor(A, B)`

<a href=https://onnx.ai/onnx/operators/onnx__Xor.html><i>ONNX Documentation</i></a>
<a name="patternmatchingfunction"></a>

## pattern_matching_function
 <p><i>Returns the productions that match the given goal and retrieval buffers.</i></p>
<p><b>pattern_matching_function(productions, goal, retrieval)</b> = actr.pattern_matching_function(productions,goal,retrieval)</p>

Python version: `actr.pattern_matching_function(productions,goal,retrieval)`

<a name="patterntostring"></a>

## pattern_to_string
 <p><i>Converts a pattern dictionary to a string format.</i></p>
<p><b>pattern_to_string(chunk)</b> = actr.pattern_to_string(chunk)</p>

Python version: `actr.pattern_to_string(chunk)`

<a name="retrievechunk"></a>

## retrieve_chunk
 <p><i>Retrieve a chunk from declarative memory given a pattern.</i></p>
<p><b>retrieve_chunk(pattern, dm_chunks, types)</b> = actr.retrieve_chunk(pattern,dm_chunks,types)</p>

Python version: `actr.retrieve_chunk(pattern,dm_chunks,types)`

<a name="sin"></a>

## sin
 <p><i>Sine function</i></p>
<p><b>sin(variable0, scale)</b> = scale * sin(variable0)</p>

Python version: `scale * numpy.sin(variable0)`

<a name="sinh"></a>

## sinh
 <p><i>Hyperbolic sine function</i></p>
<p><b>sinh(variable0, scale)</b> = scale * sinh(variable0)</p>

Python version: `scale * numpy.sinh(variable0)`

<a name="tan"></a>

## tan
 <p><i>Tangent function</i></p>
<p><b>tan(variable0, scale)</b> = scale * tan(variable0)</p>

Python version: `scale * numpy.tan(variable0)`

<a name="tanh"></a>

## tanh
 <p><i>Hyperbolic tangent function</i></p>
<p><b>tanh(variable0, scale)</b> = scale * tanh(variable0)</p>

Python version: `scale * numpy.tanh(variable0)`

<a name="updatebuffer"></a>

## update_buffer
 <p><i>Returns a pattern to update the given buffer with.</i></p>
<p><b>update_buffer(production, buffer)</b> = actr.update_buffer(production,buffer)</p>

Python version: `actr.update_buffer(production,buffer)`

<a name="updategoal"></a>

## update_goal
 <p><i>Returns a pattern to update the goal buffer with.</i></p>
<p><b>update_goal(production)</b> = actr.update_goal(production)</p>

Python version: `actr.update_goal(production)`

<a name="updateretrieval"></a>

## update_retrieval
 <p><i>Returns a pattern to update the retrieval buffer with.</i></p>
<p><b>update_retrieval(production)</b> = actr.update_retrieval(production)</p>

Python version: `actr.update_retrieval(production)`
