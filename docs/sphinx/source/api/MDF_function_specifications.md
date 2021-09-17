# Specification of standard functions in ModECI v0.2
**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**
These functions are defined in https://github.com/ModECI/MDF/blob/main/src/modeci_mdf/standard_functions.py
## All functions:
 | <a href="#matmul">MatMul</a> | <a href="#relu">Relu</a> | <a href="#change_goal">change_goal</a> | <a href="#check_termination">check_termination</a> | <a href="#conflict_resolution_function">conflict_resolution_function</a> | <a href="#cos">cos</a> | <a href="#exponential">exponential</a> | <a href="#linear">linear</a> | <a href="#logistic">logistic</a> | <a href="#onnxabs">onnx::Abs</a> | <a href="#onnxacos">onnx::Acos</a> | <a href="#onnxacosh">onnx::Acosh</a> | <a href="#onnxadd">onnx::Add</a> | <a href="#onnxand">onnx::And</a> | <a href="#onnxargmax">onnx::ArgMax</a> | <a href="#onnxargmin">onnx::ArgMin</a> | <a href="#onnxasin">onnx::Asin</a> | <a href="#onnxasinh">onnx::Asinh</a> | <a href="#onnxatan">onnx::Atan</a> | <a href="#onnxatanh">onnx::Atanh</a> | <a href="#onnxaveragepool">onnx::AveragePool</a> | <a href="#onnxbatchnormalization">onnx::BatchNormalization</a> | <a href="#onnxbitshift">onnx::BitShift</a> | <a href="#onnxcast">onnx::Cast</a> | <a href="#onnxceil">onnx::Ceil</a> | <a href="#onnxcelu">onnx::Celu</a> | <a href="#onnxclip">onnx::Clip</a> | <a href="#onnxcompress">onnx::Compress</a> | <a href="#onnxconcat">onnx::Concat</a> | <a href="#onnxconcatfromsequence">onnx::ConcatFromSequence</a> | <a href="#onnxconstant">onnx::Constant</a> | <a href="#onnxconstantofshape">onnx::ConstantOfShape</a> | <a href="#onnxconv">onnx::Conv</a> | <a href="#onnxconvinteger">onnx::ConvInteger</a> | <a href="#onnxconvtranspose">onnx::ConvTranspose</a> | <a href="#onnxcos">onnx::Cos</a> | <a href="#onnxcosh">onnx::Cosh</a> | <a href="#onnxcumsum">onnx::CumSum</a> | <a href="#onnxdepthtospace">onnx::DepthToSpace</a> | <a href="#onnxdequantizelinear">onnx::DequantizeLinear</a> | <a href="#onnxdet">onnx::Det</a> | <a href="#onnxdiv">onnx::Div</a> | <a href="#onnxdropout">onnx::Dropout</a> | <a href="#onnxdynamicquantizelinear">onnx::DynamicQuantizeLinear</a> | <a href="#onnxeinsum">onnx::Einsum</a> | <a href="#onnxelu">onnx::Elu</a> | <a href="#onnxequal">onnx::Equal</a> | <a href="#onnxerf">onnx::Erf</a> | <a href="#onnxexp">onnx::Exp</a> | <a href="#onnxexpand">onnx::Expand</a> | <a href="#onnxeyelike">onnx::EyeLike</a> | <a href="#onnxflatten">onnx::Flatten</a> | <a href="#onnxfloor">onnx::Floor</a> | <a href="#onnxgru">onnx::GRU</a> | <a href="#onnxgather">onnx::Gather</a> | <a href="#onnxgatherelements">onnx::GatherElements</a> | <a href="#onnxgathernd">onnx::GatherND</a> | <a href="#onnxgemm">onnx::Gemm</a> | <a href="#onnxglobalaveragepool">onnx::GlobalAveragePool</a> | <a href="#onnxgloballppool">onnx::GlobalLpPool</a> | <a href="#onnxglobalmaxpool">onnx::GlobalMaxPool</a> | <a href="#onnxgreater">onnx::Greater</a> | <a href="#onnxgreaterorequal">onnx::GreaterOrEqual</a> | <a href="#onnxhardsigmoid">onnx::HardSigmoid</a> | <a href="#onnxhardmax">onnx::Hardmax</a> | <a href="#onnxidentity">onnx::Identity</a> | <a href="#onnxif">onnx::If</a> | <a href="#onnxinstancenormalization">onnx::InstanceNormalization</a> | <a href="#onnxisinf">onnx::IsInf</a> | <a href="#onnxisnan">onnx::IsNaN</a> | <a href="#onnxlrn">onnx::LRN</a> | <a href="#onnxlstm">onnx::LSTM</a> | <a href="#onnxleakyrelu">onnx::LeakyRelu</a> | <a href="#onnxless">onnx::Less</a> | <a href="#onnxlessorequal">onnx::LessOrEqual</a> | <a href="#onnxlog">onnx::Log</a> | <a href="#onnxlogsoftmax">onnx::LogSoftmax</a> | <a href="#onnxloop">onnx::Loop</a> | <a href="#onnxlpnormalization">onnx::LpNormalization</a> | <a href="#onnxlppool">onnx::LpPool</a> | <a href="#onnxmatmul">onnx::MatMul</a> | <a href="#onnxmatmulinteger">onnx::MatMulInteger</a> | <a href="#onnxmax">onnx::Max</a> | <a href="#onnxmaxpool">onnx::MaxPool</a> | <a href="#onnxmaxroipool">onnx::MaxRoiPool</a> | <a href="#onnxmaxunpool">onnx::MaxUnpool</a> | <a href="#onnxmean">onnx::Mean</a> | <a href="#onnxmeanvariancenormalization">onnx::MeanVarianceNormalization</a> | <a href="#onnxmin">onnx::Min</a> | <a href="#onnxmod">onnx::Mod</a> | <a href="#onnxmul">onnx::Mul</a> | <a href="#onnxmultinomial">onnx::Multinomial</a> | <a href="#onnxneg">onnx::Neg</a> | <a href="#onnxnegativeloglikelihoodloss">onnx::NegativeLogLikelihoodLoss</a> | <a href="#onnxnonmaxsuppression">onnx::NonMaxSuppression</a> | <a href="#onnxnonzero">onnx::NonZero</a> | <a href="#onnxnot">onnx::Not</a> | <a href="#onnxonehot">onnx::OneHot</a> | <a href="#onnxor">onnx::Or</a> | <a href="#onnxprelu">onnx::PRelu</a> | <a href="#onnxpad">onnx::Pad</a> | <a href="#onnxpow">onnx::Pow</a> | <a href="#onnxqlinearconv">onnx::QLinearConv</a> | <a href="#onnxqlinearmatmul">onnx::QLinearMatMul</a> | <a href="#onnxquantizelinear">onnx::QuantizeLinear</a> | <a href="#onnxrnn">onnx::RNN</a> | <a href="#onnxrandomnormal">onnx::RandomNormal</a> | <a href="#onnxrandomnormallike">onnx::RandomNormalLike</a> | <a href="#onnxrandomuniform">onnx::RandomUniform</a> | <a href="#onnxrandomuniformlike">onnx::RandomUniformLike</a> | <a href="#onnxrange">onnx::Range</a> | <a href="#onnxreciprocal">onnx::Reciprocal</a> | <a href="#onnxreducel1">onnx::ReduceL1</a> | <a href="#onnxreducel2">onnx::ReduceL2</a> | <a href="#onnxreducelogsum">onnx::ReduceLogSum</a> | <a href="#onnxreducelogsumexp">onnx::ReduceLogSumExp</a> | <a href="#onnxreducemax">onnx::ReduceMax</a> | <a href="#onnxreducemean">onnx::ReduceMean</a> | <a href="#onnxreducemin">onnx::ReduceMin</a> | <a href="#onnxreduceprod">onnx::ReduceProd</a> | <a href="#onnxreducesum">onnx::ReduceSum</a> | <a href="#onnxreducesumsquare">onnx::ReduceSumSquare</a> | <a href="#onnxrelu">onnx::Relu</a> | <a href="#onnxreshape">onnx::Reshape</a> | <a href="#onnxresize">onnx::Resize</a> | <a href="#onnxreversesequence">onnx::ReverseSequence</a> | <a href="#onnxroialign">onnx::RoiAlign</a> | <a href="#onnxround">onnx::Round</a> | <a href="#onnxscan">onnx::Scan</a> | <a href="#onnxscatter">onnx::Scatter</a> | <a href="#onnxscatterelements">onnx::ScatterElements</a> | <a href="#onnxscatternd">onnx::ScatterND</a> | <a href="#onnxselu">onnx::Selu</a> | <a href="#onnxsequenceat">onnx::SequenceAt</a> | <a href="#onnxsequenceconstruct">onnx::SequenceConstruct</a> | <a href="#onnxsequenceempty">onnx::SequenceEmpty</a> | <a href="#onnxsequenceerase">onnx::SequenceErase</a> | <a href="#onnxsequenceinsert">onnx::SequenceInsert</a> | <a href="#onnxsequencelength">onnx::SequenceLength</a> | <a href="#onnxshape">onnx::Shape</a> | <a href="#onnxshrink">onnx::Shrink</a> | <a href="#onnxsigmoid">onnx::Sigmoid</a> | <a href="#onnxsign">onnx::Sign</a> | <a href="#onnxsin">onnx::Sin</a> | <a href="#onnxsinh">onnx::Sinh</a> | <a href="#onnxsize">onnx::Size</a> | <a href="#onnxslice">onnx::Slice</a> | <a href="#onnxsoftmax">onnx::Softmax</a> | <a href="#onnxsoftmaxcrossentropyloss">onnx::SoftmaxCrossEntropyLoss</a> | <a href="#onnxsoftplus">onnx::Softplus</a> | <a href="#onnxsoftsign">onnx::Softsign</a> | <a href="#onnxspacetodepth">onnx::SpaceToDepth</a> | <a href="#onnxsplit">onnx::Split</a> | <a href="#onnxsplittosequence">onnx::SplitToSequence</a> | <a href="#onnxsqrt">onnx::Sqrt</a> | <a href="#onnxsqueeze">onnx::Squeeze</a> | <a href="#onnxstringnormalizer">onnx::StringNormalizer</a> | <a href="#onnxsub">onnx::Sub</a> | <a href="#onnxsum">onnx::Sum</a> | <a href="#onnxtan">onnx::Tan</a> | <a href="#onnxtanh">onnx::Tanh</a> | <a href="#onnxtfidfvectorizer">onnx::TfIdfVectorizer</a> | <a href="#onnxthresholdedrelu">onnx::ThresholdedRelu</a> | <a href="#onnxtile">onnx::Tile</a> | <a href="#onnxtopk">onnx::TopK</a> | <a href="#onnxtranspose">onnx::Transpose</a> | <a href="#onnxunique">onnx::Unique</a> | <a href="#onnxunsqueeze">onnx::Unsqueeze</a> | <a href="#onnxupsample">onnx::Upsample</a> | <a href="#onnxwhere">onnx::Where</a> | <a href="#onnxxor">onnx::Xor</a> | <a href="#pattern_matching_function">pattern_matching_function</a> | <a href="#retrieve_chunk">retrieve_chunk</a> | <a href="#sin">sin</a> | <a href="#update_goal">update_goal</a> | <a href="#update_retrieval">update_retrieval</a> | 
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

## onnx::LessOrEqual
 <p><i>
Returns the tensor resulted from performing the `less_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::LessOrEqual(A, B)</b> = onnx_ops.lessorequal(A, B)</p> 
<p>Python version: onnx_ops.lessorequal(A, B)</p> 

## onnx::Celu
 <p><i>
Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```
</i></p> 
<p><b>onnx::Celu(X)</b> = onnx_ops.celu(X, alpha)</p> 
<p>Python version: onnx_ops.celu(X, alpha)</p> 

## onnx::ConcatFromSequence
 <p><i>
Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.
</i></p> 
<p><b>onnx::ConcatFromSequence(input_sequence)</b> = onnx_ops.concatfromsequence(input_sequence, axis, new_axis)</p> 
<p>Python version: onnx_ops.concatfromsequence(input_sequence, axis, new_axis)</p> 

## onnx::SequenceAt
 <p><i>
Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
</i></p> 
<p><b>onnx::SequenceAt(input_sequence, position)</b> = onnx_ops.sequenceat(input_sequence, position)</p> 
<p>Python version: onnx_ops.sequenceat(input_sequence, position)</p> 

## onnx::SequenceInsert
 <p><i>
Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
</i></p> 
<p><b>onnx::SequenceInsert(input_sequence, tensor, position)</b> = onnx_ops.sequenceinsert(input_sequence, tensor, position)</p> 
<p>Python version: onnx_ops.sequenceinsert(input_sequence, tensor, position)</p> 

## onnx::GatherND
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

`Example 1`

  batch_dims = 0

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  batch_dims = 0

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]

`Example 4`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]

`Example 5`

  batch_dims = 1

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[1],[0]]             # indices_shape = [2, 1]

  output  = [[2,3],[4,5]]             # output_shape = [2, 2]


</i></p> 
<p><b>onnx::GatherND(data, indices)</b> = onnx_ops.gathernd(data, indices, batch_dims)</p> 
<p>Python version: onnx_ops.gathernd(data, indices, batch_dims)</p> 

## onnx::ScatterND
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
update to a slice of the tensor.

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
<p><b>onnx::ScatterND(data, indices, updates)</b> = onnx_ops.scatternd(data, indices, updates)</p> 
<p>Python version: onnx_ops.scatternd(data, indices, updates)</p> 

## onnx::Det
 <p><i>
Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
</i></p> 
<p><b>onnx::Det(X)</b> = onnx_ops.det(X)</p> 
<p>Python version: onnx_ops.det(X)</p> 

## onnx::ScatterElements
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
<p><b>onnx::ScatterElements(data, indices, updates)</b> = onnx_ops.scatterelements(data, indices, updates, axis)</p> 
<p>Python version: onnx_ops.scatterelements(data, indices, updates, axis)</p> 

## onnx::GatherElements
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
      [
        [1, 1],
        [4, 3],
      ],
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
      [
        [4, 8, 3],
        [7, 2, 3],
      ],
  ]
```
</i></p> 
<p><b>onnx::GatherElements(data, indices)</b> = onnx_ops.gatherelements(data, indices, axis)</p> 
<p>Python version: onnx_ops.gatherelements(data, indices, axis)</p> 

## onnx::SplitToSequence
 <p><i>Split a tensor into a sequence of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into equally sized chunks(if possible).
Last chunk will be smaller if the 'input' size along the given axis 'axis' is not divisible
by 'split'.
Otherwise, the tensor is split into 'size(split)' chunks, with lengths of the parts on 'axis'
specified in 'split'. In this scenario, the sum of entries in 'split' must be equal to the
dimension size of input tensor on 'axis'.
</i></p> 
<p><b>onnx::SplitToSequence(input, split)</b> = onnx_ops.splittosequence(input, split, axis, keepdims)</p> 
<p>Python version: onnx_ops.splittosequence(input, split, axis, keepdims)</p> 

## onnx::DynamicQuantizeLinear
 <p><i>
A Function to fuse calculation for Scale, Zero Point and FP32->8Bit convertion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
 y_scale = (max(x) - min(x))/(qmax - qmin)
 * where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
 * data range is adjusted to include 0.
```
Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
```
Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.
```
</i></p> 
<p><b>onnx::DynamicQuantizeLinear(x)</b> = onnx_ops.dynamicquantizelinear(x)</p> 
<p>Python version: onnx_ops.dynamicquantizelinear(x)</p> 

## onnx::Round
 <p><i>
Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halfs, the rule is to round them to the nearest even integer.
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
<p><b>onnx::Round(X)</b> = onnx_ops.round(X)</p> 
<p>Python version: onnx_ops.round(X)</p> 

## onnx::CumSum
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
<p><b>onnx::CumSum(x, axis)</b> = onnx_ops.cumsum(x, axis, exclusive, reverse)</p> 
<p>Python version: onnx_ops.cumsum(x, axis, exclusive, reverse)</p> 

## onnx::BitShift
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
<p><b>onnx::BitShift(X, Y)</b> = onnx_ops.bitshift(X, Y, direction)</p> 
<p>Python version: onnx_ops.bitshift(X, Y, direction)</p> 

## onnx::RoiAlign
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
<p><b>onnx::RoiAlign(X, rois, batch_indices)</b> = onnx_ops.roialign(X, rois, batch_indices, mode, output_height, output_width, sampling_ratio, spatial_scale)</p> 
<p>Python version: onnx_ops.roialign(X, rois, batch_indices, mode, output_height, output_width, sampling_ratio, spatial_scale)</p> 

## onnx::ReverseSequence
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
<p><b>onnx::ReverseSequence(input, sequence_lens)</b> = onnx_ops.reversesequence(input, sequence_lens, batch_axis, time_axis)</p> 
<p>Python version: onnx_ops.reversesequence(input, sequence_lens, batch_axis, time_axis)</p> 

## onnx::NonMaxSuppression
 <p><i>
Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.
</i></p> 
<p><b>onnx::NonMaxSuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)</b> = onnx_ops.nonmaxsuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box)</p> 
<p>Python version: onnx_ops.nonmaxsuppression(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, center_point_box)</p> 

## onnx::IsInf
 <p><i>Map infinity to true and other values to false.</i></p> 
<p><b>onnx::IsInf(X)</b> = onnx_ops.isinf(X, detect_negative, detect_positive)</p> 
<p>Python version: onnx_ops.isinf(X, detect_negative, detect_positive)</p> 

## onnx::QuantizeLinear
 <p><i>
The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor. The scale factor can be a scalar
(per-tensor/layer quantization), or a 1-D tensor for per-axis quantization. The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.
</i></p> 
<p><b>onnx::QuantizeLinear(x, y_scale, y_zero_point)</b> = onnx_ops.quantizelinear(x, y_scale, y_zero_point, axis)</p> 
<p>Python version: onnx_ops.quantizelinear(x, y_scale, y_zero_point, axis)</p> 

## onnx::QLinearConv
 <p><i>
The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
When bias is present it must be quantized using scale = input scale * weight scale and
zero point as 0.
</i></p> 
<p><b>onnx::QLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B)</b> = onnx_ops.qlinearconv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, auto_pad, dilations, group, kernel_shape, pads, strides)</p> 
<p>Python version: onnx_ops.qlinearconv(x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, B, auto_pad, dilations, group, kernel_shape, pads, strides)</p> 

## onnx::ConvInteger
 <p><i>
The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
</i></p> 
<p><b>onnx::ConvInteger(x, w, x_zero_point, w_zero_point)</b> = onnx_ops.convinteger(x, w, x_zero_point, w_zero_point, auto_pad, dilations, group, kernel_shape, pads, strides)</p> 
<p>Python version: onnx_ops.convinteger(x, w, x_zero_point, w_zero_point, auto_pad, dilations, group, kernel_shape, pads, strides)</p> 

## onnx::QLinearMatMul
 <p><i>
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output.
The quantization formula is y = saturate((x / y_scale) + y_zero_point). For (x / y_scale), it is rounding to nearest ties to even.
Refer to https://en.wikipedia.org/wiki/Rounding for details. Scale and zero point must have same shape.
They must be either scalar (per tensor) or 1-D tensor (per row for 'a' and per column for 'b'). If scale and zero point are 1-D tensor,
the number of elements of scale and zero point tensor of input 'a' and output 'y' should be equal to the number of rows of input 'a',
and the number of elements of scale and zero point tensor of input 'b' should be equal to the number of columns of input 'b'.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
</i></p> 
<p><b>onnx::QLinearMatMul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point)</b> = onnx_ops.qlinearmatmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point)</p> 
<p>Python version: onnx_ops.qlinearmatmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point)</p> 

## onnx::MatMulInteger
 <p><i>
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
</i></p> 
<p><b>onnx::MatMulInteger(A, B, a_zero_point, b_zero_point)</b> = onnx_ops.matmulinteger(A, B, a_zero_point, b_zero_point)</p> 
<p>Python version: onnx_ops.matmulinteger(A, B, a_zero_point, b_zero_point)</p> 

## onnx::StringNormalizer
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
<p><b>onnx::StringNormalizer(X)</b> = onnx_ops.stringnormalizer(X, case_change_action, is_case_sensitive, locale, stopwords)</p> 
<p>Python version: onnx_ops.stringnormalizer(X, case_change_action, is_case_sensitive, locale, stopwords)</p> 

## onnx::MeanVarianceNormalization
 <p><i>
      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
</i></p> 
<p><b>onnx::MeanVarianceNormalization(X)</b> = onnx_ops.meanvariancenormalization(X, axes)</p> 
<p>Python version: onnx_ops.meanvariancenormalization(X, axes)</p> 

## onnx::TfIdfVectorizer
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
<p><b>onnx::TfIdfVectorizer(X)</b> = onnx_ops.tfidfvectorizer(X, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights)</p> 
<p>Python version: onnx_ops.tfidfvectorizer(X, max_gram_length, max_skip_count, min_gram_length, mode, ngram_counts, ngram_indexes, pool_int64s, pool_strings, weights)</p> 

## onnx::Range
 <p><i>
Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below-

`number_of_elements = max( ceil( (limit - start) / delta ) , 0 )`

The pseudocode determining the contents of the output is shown below-

`for(int i=0; i<number_of_elements; ++i)`

`{`

`    output[i] =  start + (i * delta);  `

`}`

`Example 1`
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]

`Example 2`
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]

</i></p> 
<p><b>onnx::Range(start, limit, delta)</b> = onnx_ops.range(start, limit, delta)</p> 
<p>Python version: onnx_ops.range(start, limit, delta)</p> 

## onnx::NonZero
 <p><i>
    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
    NonZero behaves similar to numpy.nonzero:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
</i></p> 
<p><b>onnx::NonZero(X)</b> = onnx_ops.nonzero(X)</p> 
<p>Python version: onnx_ops.nonzero(X)</p> 

## onnx::Sign
 <p><i>
Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
</i></p> 
<p><b>onnx::Sign(input)</b> = onnx_ops.sign(input)</p> 
<p>Python version: onnx_ops.sign(input)</p> 

## onnx::IsNaN
 <p><i>Returns which elements of the input are NaN.</i></p> 
<p><b>onnx::IsNaN(X)</b> = onnx_ops.isnan(X)</p> 
<p>Python version: onnx_ops.isnan(X)</p> 

## onnx::SequenceErase
 <p><i>
Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.
</i></p> 
<p><b>onnx::SequenceErase(input_sequence, position)</b> = onnx_ops.sequenceerase(input_sequence, position)</p> 
<p>Python version: onnx_ops.sequenceerase(input_sequence, position)</p> 

## onnx::Shrink
 <p><i>
Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.
</i></p> 
<p><b>onnx::Shrink(input)</b> = onnx_ops.shrink(input, bias, lambd)</p> 
<p>Python version: onnx_ops.shrink(input, bias, lambd)</p> 

## onnx::Sinh
 <p><i>
Calculates the hyperbolic sine of the given input tensor element-wise.
</i></p> 
<p><b>onnx::Sinh(input)</b> = onnx_ops.sinh(input)</p> 
<p>Python version: onnx_ops.sinh(input)</p> 

## onnx::Mod
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
<p><b>onnx::Mod(A, B)</b> = onnx_ops.mod(A, B, fmod)</p> 
<p>Python version: onnx_ops.mod(A, B, fmod)</p> 

## onnx::Scatter
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
<p><b>onnx::Scatter(data, indices, updates)</b> = onnx_ops.scatter(data, indices, updates, axis)</p> 
<p>Python version: onnx_ops.scatter(data, indices, updates, axis)</p> 

## onnx::OneHot
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
<p><b>onnx::OneHot(indices, depth, values)</b> = onnx_ops.onehot(indices, depth, values, axis)</p> 
<p>Python version: onnx_ops.onehot(indices, depth, values, axis)</p> 

## onnx::MaxUnpool
 <p><i>
MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corrsponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corrsponding
 pooling op that the unpooling op is trying to invert.
</i></p> 
<p><b>onnx::MaxUnpool(X, I, output_shape)</b> = onnx_ops.maxunpool(X, I, output_shape, kernel_shape, pads, strides)</p> 
<p>Python version: onnx_ops.maxunpool(X, I, output_shape, kernel_shape, pads, strides)</p> 

## onnx::EyeLike
 <p><i>
Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
</i></p> 
<p><b>onnx::EyeLike(input)</b> = onnx_ops.eyelike(input, dtype, k)</p> 
<p>Python version: onnx_ops.eyelike(input, dtype, k)</p> 

## onnx::ConstantOfShape
 <p><i>
Generate a tensor with given value and shape.
</i></p> 
<p><b>onnx::ConstantOfShape(input)</b> = onnx_ops.constantofshape(input, value)</p> 
<p>Python version: onnx_ops.constantofshape(input, value)</p> 

## onnx::Compress
 <p><i>
    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    </i></p> 
<p><b>onnx::Compress(input, condition)</b> = onnx_ops.compress(input, condition, axis)</p> 
<p>Python version: onnx_ops.compress(input, condition, axis)</p> 

## onnx::Scan
 <p><i>
Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }

</i></p> 
<p><b>onnx::Scan(initial_state_and_scan_inputs)</b> = onnx_ops.scan(initial_state_and_scan_inputs, body, num_scan_inputs, scan_input_axes, scan_input_directions, scan_output_axes, scan_output_directions)</p> 
<p>Python version: onnx_ops.scan(initial_state_and_scan_inputs, body, num_scan_inputs, scan_input_axes, scan_input_directions, scan_output_axes, scan_output_directions)</p> 

## onnx::DequantizeLinear
 <p><i>
The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantizations.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
</i></p> 
<p><b>onnx::DequantizeLinear(x, x_scale, x_zero_point)</b> = onnx_ops.dequantizelinear(x, x_scale, x_zero_point, axis)</p> 
<p>Python version: onnx_ops.dequantizelinear(x, x_scale, x_zero_point, axis)</p> 

## onnx::ThresholdedRelu
 <p><i>
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
</i></p> 
<p><b>onnx::ThresholdedRelu(X)</b> = onnx_ops.thresholdedrelu(X, alpha)</p> 
<p>Python version: onnx_ops.thresholdedrelu(X, alpha)</p> 

## onnx::Expand
 <p><i>
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimension must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
</i></p> 
<p><b>onnx::Expand(input, shape)</b> = onnx_ops.expand(input, shape)</p> 
<p>Python version: onnx_ops.expand(input, shape)</p> 

## onnx::Multinomial
 <p><i>
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
</i></p> 
<p><b>onnx::Multinomial(input)</b> = onnx_ops.multinomial(input, dtype, sample_size, seed)</p> 
<p>Python version: onnx_ops.multinomial(input, dtype, sample_size, seed)</p> 

## onnx::Asin
 <p><i>
Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
</i></p> 
<p><b>onnx::Asin(input)</b> = onnx_ops.asin(input)</p> 
<p>Python version: onnx_ops.amath.sin(input)</p> 

## onnx::Xor
 <p><i>
Returns the tensor resulted from performing the `xor` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Xor(A, B)</b> = onnx_ops.xor(A, B)</p> 
<p>Python version: onnx_ops.xor(A, B)</p> 

## onnx::Einsum
 <p><i>
An einsum of the form ```term1, term2 -> output-term``` produces an output tensor using the following equation

```output[output-term] = reduce-sum( input1[term1] * input2[term] )```

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
<p><b>onnx::Einsum(Inputs)</b> = onnx_ops.einsum(Inputs, equation)</p> 
<p>Python version: onnx_ops.einsum(Inputs, equation)</p> 

## onnx::Floor
 <p><i>
Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise.
</i></p> 
<p><b>onnx::Floor(X)</b> = onnx_ops.floor(X)</p> 
<p>Python version: onnx_ops.floor(X)</p> 

## onnx::ReduceSumSquare
 <p><i>
Computes the sum square of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceSumSquare(data)</b> = onnx_ops.reducesumsquare(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reducesumsquare(data, axes, keepdims)</p> 

## onnx::Upsample
 <p><i>
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
</i></p> 
<p><b>onnx::Upsample(X, scales)</b> = onnx_ops.upsample(X, scales, mode)</p> 
<p>Python version: onnx_ops.upsample(X, scales, mode)</p> 

## onnx::And
 <p><i>
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::And(A, B)</b> = onnx_ops.and(A, B)</p> 
<p>Python version: onnx_ops.and(A, B)</p> 

## onnx::Tile
 <p><i>Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
</i></p> 
<p><b>onnx::Tile(input, repeats)</b> = onnx_ops.tile(input, repeats)</p> 
<p>Python version: onnx_ops.tile(input, repeats)</p> 

## onnx::Sub
 <p><i>
Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Sub(A, B)</b> = onnx_ops.sub(A, B)</p> 
<p>Python version: onnx_ops.sub(A, B)</p> 

## onnx::Squeeze
 <p><i>
Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
</i></p> 
<p><b>onnx::Squeeze(data, axes)</b> = onnx_ops.squeeze(data, axes)</p> 
<p>Python version: onnx_ops.squeeze(data, axes)</p> 

## onnx::Acosh
 <p><i>
Calculates the hyperbolic arccosine of the given input tensor element-wise.
</i></p> 
<p><b>onnx::Acosh(input)</b> = onnx_ops.acosh(input)</p> 
<p>Python version: onnx_ops.acosh(input)</p> 

## onnx::ReduceLogSum
 <p><i>
Computes the log sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceLogSum(data)</b> = onnx_ops.reducelogsum(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reducelogsum(data, axes, keepdims)</p> 

## onnx::Split
 <p><i>Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using input 'split'.
Otherwise, the tensor is split to equal sized parts.
</i></p> 
<p><b>onnx::Split(input, split)</b> = onnx_ops.split(input, split, axis)</p> 
<p>Python version: onnx_ops.split(input, split, axis)</p> 

## onnx::Where
 <p><i>
    Return elements, either from X or Y, depending on condition
    (with Numpy-style broadcasting support).
    Where behaves like numpy.where with three parameters:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
</i></p> 
<p><b>onnx::Where(condition, X, Y)</b> = onnx_ops.where(condition, X, Y)</p> 
<p>Python version: onnx_ops.where(condition, X, Y)</p> 

## onnx::Sqrt
 <p><i>
Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.
</i></p> 
<p><b>onnx::Sqrt(X)</b> = onnx_ops.sqrt(X)</p> 
<p>Python version: onnx_ops.sqrt(X)</p> 

## onnx::Softsign
 <p><i>
Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
</i></p> 
<p><b>onnx::Softsign(input)</b> = onnx_ops.softsign(input)</p> 
<p>Python version: onnx_ops.softsign(input)</p> 

## onnx::Softplus
 <p><i>
Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
</i></p> 
<p><b>onnx::Softplus(X)</b> = onnx_ops.softplus(X)</p> 
<p>Python version: onnx_ops.softplus(X)</p> 

## onnx::Cos
 <p><i>
Calculates the cosine of the given input tensor, element-wise.
</i></p> 
<p><b>onnx::Cos(input)</b> = onnx_ops.cos(input)</p> 
<p>Python version: onnx_ops.math.cos(input)</p> 

## onnx::SpaceToDepth
 <p><i>SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
</i></p> 
<p><b>onnx::SpaceToDepth(input)</b> = onnx_ops.spacetodepth(input, blocksize)</p> 
<p>Python version: onnx_ops.spacetodepth(input, blocksize)</p> 

## onnx::GreaterOrEqual
 <p><i>
Returns the tensor resulted from performing the `greater_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::GreaterOrEqual(A, B)</b> = onnx_ops.greaterorequal(A, B)</p> 
<p>Python version: onnx_ops.greaterorequal(A, B)</p> 

## onnx::Softmax
 <p><i>
The operator computes the normalized exponential values for the given input:

 Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) 

The input does not need to explicitly be a 2D vector. The "axis" attribute
indicates the dimension along which Softmax will be performed.
The output tensor has the same shape
and contains the Softmax values of the corresponding input.
</i></p> 
<p><b>onnx::Softmax(input)</b> = onnx_ops.softmax(input, axis)</p> 
<p>Python version: onnx_ops.softmax(input, axis)</p> 

## onnx::Erf
 <p><i>
Computes the error function of the given input tensor element-wise.
</i></p> 
<p><b>onnx::Erf(input)</b> = onnx_ops.erf(input)</p> 
<p>Python version: onnx_ops.erf(input)</p> 

## onnx::Size
 <p><i>
Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.
</i></p> 
<p><b>onnx::Size(data)</b> = onnx_ops.size(data)</p> 
<p>Python version: onnx_ops.size(data)</p> 

## onnx::Max
 <p><i>
Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Max(data_0)</b> = onnx_ops.max(data_0)</p> 
<p>Python version: onnx_ops.max(data_0)</p> 

## onnx::Tanh
 <p><i>
Calculates the hyperbolic tangent of the given input tensor element-wise.
</i></p> 
<p><b>onnx::Tanh(input)</b> = onnx_ops.tanh(input)</p> 
<p>Python version: onnx_ops.tanh(input)</p> 

## onnx::Transpose
 <p><i>
Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
</i></p> 
<p><b>onnx::Transpose(data)</b> = onnx_ops.transpose(data, perm)</p> 
<p>Python version: onnx_ops.transpose(data, perm)</p> 

## onnx::Shape
 <p><i>
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
</i></p> 
<p><b>onnx::Shape(data)</b> = onnx_ops.shape(data)</p> 
<p>Python version: onnx_ops.shape(data)</p> 

## onnx::Selu
 <p><i>
Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
</i></p> 
<p><b>onnx::Selu(X)</b> = onnx_ops.selu(X, alpha, gamma)</p> 
<p>Python version: onnx_ops.selu(X, alpha, gamma)</p> 

## onnx::Sum
 <p><i>
Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Sum(data_0)</b> = onnx_ops.sum(data_0)</p> 
<p>Python version: onnx_ops.sum(data_0)</p> 

## onnx::Relu
 <p><i>
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.
</i></p> 
<p><b>onnx::Relu(X)</b> = onnx_ops.relu(X)</p> 
<p>Python version: onnx_ops.relu(X)</p> 

## onnx::NegativeLogLikelihoodLoss
 <p><i>
A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].

When an optional "weight" is provided, the sample loss is calculated as:

    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].

loss is zero for the case when target-value equals ignore_index.

    loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index

If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:

    mean(loss), if "weight" is not provided,

or if weight is provided,

    sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.

If "reduction" attribute is set to "sum", the output is a scalar:
    sum(loss).

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

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

Example 2:

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

Example 3:

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
</i></p> 
<p><b>onnx::NegativeLogLikelihoodLoss(input, target, weight)</b> = onnx_ops.negativeloglikelihoodloss(input, target, weight, ignore_index, reduction)</p> 
<p>Python version: onnx_ops.negativeloglikelihoodloss(input, target, weight, ignore_index, reduction)</p> 

## onnx::SequenceLength
 <p><i>
Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
</i></p> 
<p><b>onnx::SequenceLength(input_sequence)</b> = onnx_ops.sequencelength(input_sequence)</p> 
<p>Python version: onnx_ops.sequencelength(input_sequence)</p> 

## onnx::ReduceMin
 <p><i>
Computes the min of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceMin(data)</b> = onnx_ops.reducemin(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reducemin(data, axes, keepdims)</p> 

## onnx::ReduceL1
 <p><i>
Computes the L1 norm of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceL1(data)</b> = onnx_ops.reducel1(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reducel1(data, axes, keepdims)</p> 

## onnx::Reciprocal
 <p><i>
Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.
</i></p> 
<p><b>onnx::Reciprocal(X)</b> = onnx_ops.reciprocal(X)</p> 
<p>Python version: onnx_ops.reciprocal(X)</p> 

## onnx::Mul
 <p><i>
Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Mul(A, B)</b> = onnx_ops.mul(A, B)</p> 
<p>Python version: onnx_ops.mul(A, B)</p> 

## onnx::RandomUniformLike
 <p><i>
Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
</i></p> 
<p><b>onnx::RandomUniformLike(input)</b> = onnx_ops.randomuniformlike(input, dtype, high, low, seed)</p> 
<p>Python version: onnx_ops.randomuniformlike(input, dtype, high, low, seed)</p> 

## onnx::Sin
 <p><i>
Calculates the sine of the given input tensor, element-wise.
</i></p> 
<p><b>onnx::Sin(input)</b> = onnx_ops.sin(input)</p> 
<p>Python version: onnx_ops.math.sin(input)</p> 

## onnx::Sigmoid
 <p><i>
Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.
</i></p> 
<p><b>onnx::Sigmoid(X)</b> = onnx_ops.sigmoid(X)</p> 
<p>Python version: onnx_ops.sigmoid(X)</p> 

## onnx::RandomNormalLike
 <p><i>
Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
</i></p> 
<p><b>onnx::RandomNormalLike(input)</b> = onnx_ops.randomnormallike(input, dtype, mean, scale, seed)</p> 
<p>Python version: onnx_ops.randomnormallike(input, dtype, mean, scale, seed)</p> 

## onnx::Asinh
 <p><i>
Calculates the hyperbolic arcsine of the given input tensor element-wise.
</i></p> 
<p><b>onnx::Asinh(input)</b> = onnx_ops.asinh(input)</p> 
<p>Python version: onnx_ops.asinh(input)</p> 

## onnx::RNN
 <p><i>
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p> 
<p><b>onnx::RNN(X, W, R, B, sequence_lens, initial_h)</b> = onnx_ops.rnn(X, W, R, B, sequence_lens, initial_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size)</p> 
<p>Python version: onnx_ops.rnn(X, W, R, B, sequence_lens, initial_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size)</p> 

## onnx::Pad
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
<p><b>onnx::Pad(data, pads, constant_value)</b> = onnx_ops.pad(data, pads, constant_value, mode)</p> 
<p>Python version: onnx_ops.pad(data, pads, constant_value, mode)</p> 

## onnx::Slice
 <p><i>
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represents number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`
when sclicing forward and 'INT_MIN' when slicing backward.
If a negative value is passed for step, it represents slicing backward.
However step value cannot be 0.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
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
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
</i></p> 
<p><b>onnx::Slice(data, starts, ends, axes, steps)</b> = onnx_ops.slice(data, starts, ends, axes, steps)</p> 
<p>Python version: onnx_ops.slice(data, starts, ends, axes, steps)</p> 

## onnx::Greater
 <p><i>
Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Greater(A, B)</b> = onnx_ops.greater(A, B)</p> 
<p>Python version: onnx_ops.greater(A, B)</p> 

## onnx::ReduceLogSumExp
 <p><i>
Computes the log sum exponent of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceLogSumExp(data)</b> = onnx_ops.reducelogsumexp(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reducelogsummath.exp(data, axes, keepdims)</p> 

## onnx::Or
 <p><i>
Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Or(A, B)</b> = onnx_ops.or(A, B)</p> 
<p>Python version: onnx_ops.or(A, B)</p> 

## onnx::Neg
 <p><i>
Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.
</i></p> 
<p><b>onnx::Neg(X)</b> = onnx_ops.neg(X)</p> 
<p>Python version: onnx_ops.neg(X)</p> 

## onnx::Mean
 <p><i>
Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Mean(data_0)</b> = onnx_ops.mean(data_0)</p> 
<p>Python version: onnx_ops.mean(data_0)</p> 

## onnx::Reshape
 <p><i>
Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor).</i></p> 
<p><b>onnx::Reshape(data, shape)</b> = onnx_ops.reshape(data, shape)</p> 
<p>Python version: onnx_ops.reshape(data, shape)</p> 

## onnx::ReduceL2
 <p><i>
Computes the L2 norm of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceL2(data)</b> = onnx_ops.reducel2(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reducel2(data, axes, keepdims)</p> 

## onnx::Flatten
 <p><i>
Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
</i></p> 
<p><b>onnx::Flatten(input)</b> = onnx_ops.flatten(input, axis)</p> 
<p>Python version: onnx_ops.flatten(input, axis)</p> 

## onnx::RandomNormal
 <p><i>
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
</i></p> 
<p><b>onnx::RandomNormal()</b> = onnx_ops.randomnormal(dtype, mean, scale, seed, shape)</p> 
<p>Python version: onnx_ops.randomnormal(dtype, mean, scale, seed, shape)</p> 

## onnx::Conv
 <p><i>
The convolution operator consumes an input tensor and a filter, and
computes the output.</i></p> 
<p><b>onnx::Conv(X, W, B)</b> = onnx_ops.conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)</p> 
<p>Python version: onnx_ops.conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides)</p> 

## onnx::GlobalMaxPool
 <p><i>
 GlobalMaxPool consumes an input tensor X and applies max pooling across
 the values in the same channel. This is equivalent to MaxPool with kernel size
 equal to the spatial dimension of input tensor.</i></p> 
<p><b>onnx::GlobalMaxPool(X)</b> = onnx_ops.globalmaxpool(X)</p> 
<p>Python version: onnx_ops.globalmaxpool(X)</p> 

## onnx::LpPool
 <p><i>
 LpPool consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing.</i></p> 
<p><b>onnx::LpPool(X)</b> = onnx_ops.lppool(X, auto_pad, kernel_shape, p, pads, strides)</p> 
<p>Python version: onnx_ops.lppool(X, auto_pad, kernel_shape, p, pads, strides)</p> 

## onnx::ReduceMax
 <p><i>
Computes the max of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceMax(data)</b> = onnx_ops.reducemax(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reducemax(data, axes, keepdims)</p> 

## onnx::Loop
 <p><i>
Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.
</i></p> 
<p><b>onnx::Loop(M, cond, v_initial)</b> = onnx_ops.loop(M, cond, v_initial, body)</p> 
<p>Python version: onnx_ops.loop(M, cond, v_initial, body)</p> 

## onnx::Log
 <p><i>
Calculates the natural log of the given input tensor, element-wise.
</i></p> 
<p><b>onnx::Log(input)</b> = onnx_ops.log(input)</p> 
<p>Python version: onnx_ops.log(input)</p> 

## onnx::LeakyRelu
 <p><i>
LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.
</i></p> 
<p><b>onnx::LeakyRelu(X)</b> = onnx_ops.leakyrelu(X, alpha)</p> 
<p>Python version: onnx_ops.leakyrelu(X, alpha)</p> 

## onnx::BatchNormalization
 <p><i>
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p> 
<p><b>onnx::BatchNormalization(X, scale, B, mean, var)</b> = onnx_ops.batchnormalization(X, scale, B, mean, var, epsilon, momentum)</p> 
<p>Python version: onnx_ops.batchnormalization(X, scale, B, mean, var, epsilon, momentum)</p> 

## onnx::Cosh
 <p><i>
Calculates the hyperbolic cosine of the given input tensor element-wise.
</i></p> 
<p><b>onnx::Cosh(input)</b> = onnx_ops.cosh(input)</p> 
<p>Python version: onnx_ops.cosh(input)</p> 

## onnx::Cast
 <p><i>
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
result 100. There are some string literals reserved for special floating-point values;
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
</i></p> 
<p><b>onnx::Cast(input)</b> = onnx_ops.cast(input, to)</p> 
<p>Python version: onnx_ops.cast(input, to)</p> 

## onnx::Not
 <p><i>
Returns the negation of the input tensor element-wise.
</i></p> 
<p><b>onnx::Not(X)</b> = onnx_ops.not(X)</p> 
<p>Python version: onnx_ops.not(X)</p> 

## onnx::LSTM
 <p><i>
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p> 
<p><b>onnx::LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, P)</b> = onnx_ops.lstm(X, W, R, B, sequence_lens, initial_h, initial_c, P, activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget)</p> 
<p>Python version: onnx_ops.lstm(X, W, R, B, sequence_lens, initial_h, initial_c, P, activation_alpha, activation_beta, activations, clip, direction, hidden_size, input_forget)</p> 

## onnx::Unsqueeze
 <p><i>
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example:
  Given an input tensor (`data`) of shape [3, 4, 5], then
  Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.

</i></p> 
<p><b>onnx::Unsqueeze(data, axes)</b> = onnx_ops.unsqueeze(data, axes)</p> 
<p>Python version: onnx_ops.unsqueeze(data, axes)</p> 

## onnx::TopK
 <p><i>
Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).

If "largest" is 1 (the default value) then the k largest elements are returned.
If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
 a tiebreaker. That is, the element with the lower index will appear first.
</i></p> 
<p><b>onnx::TopK(X, K)</b> = onnx_ops.topk(X, K, axis, largest, sorted)</p> 
<p>Python version: onnx_ops.topk(X, K, axis, largest, sorted)</p> 

## onnx::ArgMax
 <p><i>
Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulting tensor have the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the max
is selected if the max appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.</i></p> 
<p><b>onnx::ArgMax(data)</b> = onnx_ops.argmax(data, axis, keepdims, select_last_index)</p> 
<p>Python version: onnx_ops.argmax(data, axis, keepdims, select_last_index)</p> 

## onnx::LRN
 <p><i>
Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta
</i></p> 
<p><b>onnx::LRN(X)</b> = onnx_ops.lrn(X, alpha, beta, bias, size)</p> 
<p>Python version: onnx_ops.lrn(X, alpha, beta, bias, size)</p> 

## onnx::SequenceEmpty
 <p><i>
Construct an empty tensor sequence, with given data type.
</i></p> 
<p><b>onnx::SequenceEmpty()</b> = onnx_ops.sequenceempty(dtype)</p> 
<p>Python version: onnx_ops.sequenceempty(dtype)</p> 

## onnx::Acos
 <p><i>
Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
</i></p> 
<p><b>onnx::Acos(input)</b> = onnx_ops.acos(input)</p> 
<p>Python version: onnx_ops.amath.cos(input)</p> 

## onnx::RandomUniform
 <p><i>
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
</i></p> 
<p><b>onnx::RandomUniform()</b> = onnx_ops.randomuniform(dtype, high, low, seed, shape)</p> 
<p>Python version: onnx_ops.randomuniform(dtype, high, low, seed, shape)</p> 

## onnx::InstanceNormalization
 <p><i>
Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.

</i></p> 
<p><b>onnx::InstanceNormalization(input, scale, B)</b> = onnx_ops.instancenormalization(input, scale, B, epsilon)</p> 
<p>Python version: onnx_ops.instancenormalization(input, scale, B, epsilon)</p> 

## onnx::SoftmaxCrossEntropyLoss
 <p><i>Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].
</i></p> 
<p><b>onnx::SoftmaxCrossEntropyLoss(scores, labels, weights)</b> = onnx_ops.softmaxcrossentropyloss(scores, labels, weights, ignore_index, reduction)</p> 
<p>Python version: onnx_ops.softmaxcrossentropyloss(scores, labels, weights, ignore_index, reduction)</p> 

## onnx::Concat
 <p><i>Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.</i></p> 
<p><b>onnx::Concat(inputs)</b> = onnx_ops.concat(inputs, axis)</p> 
<p>Python version: onnx_ops.concat(inputs, axis)</p> 

## onnx::If
 <p><i>If conditional</i></p> 
<p><b>onnx::If(cond)</b> = onnx_ops.if(cond, else_branch, then_branch)</p> 
<p>Python version: onnx_ops.if(cond, else_branch, then_branch)</p> 

## onnx::MaxRoiPool
 <p><i>
 ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).</i></p> 
<p><b>onnx::MaxRoiPool(X, rois)</b> = onnx_ops.maxroipool(X, rois, pooled_shape, spatial_scale)</p> 
<p>Python version: onnx_ops.maxroipool(X, rois, pooled_shape, spatial_scale)</p> 

## onnx::Clip
 <p><i>
Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.
</i></p> 
<p><b>onnx::Clip(input, min, max)</b> = onnx_ops.clip(input, min, max)</p> 
<p>Python version: onnx_ops.clip(input, min, max)</p> 

## onnx::Identity
 <p><i>Identity operator</i></p> 
<p><b>onnx::Identity(input)</b> = onnx_ops.identity(input)</p> 
<p>Python version: onnx_ops.identity(input)</p> 

## onnx::ReduceProd
 <p><i>
Computes the product of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceProd(data)</b> = onnx_ops.reduceprod(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reduceprod(data, axes, keepdims)</p> 

## onnx::PRelu
 <p><i>
PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).</i></p> 
<p><b>onnx::PRelu(X, slope)</b> = onnx_ops.prelu(X, slope)</p> 
<p>Python version: onnx_ops.prelu(X, slope)</p> 

## onnx::Gather
 <p><i>
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

axis = 0 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]

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
axis = 1 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]

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
<p><b>onnx::Gather(data, indices)</b> = onnx_ops.gather(data, indices, axis)</p> 
<p>Python version: onnx_ops.gather(data, indices, axis)</p> 

## onnx::Atanh
 <p><i>
Calculates the hyperbolic arctangent of the given input tensor element-wise.
</i></p> 
<p><b>onnx::Atanh(input)</b> = onnx_ops.atanh(input)</p> 
<p>Python version: onnx_ops.atanh(input)</p> 

## onnx::HardSigmoid
 <p><i>
HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
</i></p> 
<p><b>onnx::HardSigmoid(X)</b> = onnx_ops.hardsigmoid(X, alpha, beta)</p> 
<p>Python version: onnx_ops.hardsigmoid(X, alpha, beta)</p> 

## onnx::MatMul
 <p><i>
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
</i></p> 
<p><b>onnx::MatMul(A, B)</b> = onnx_ops.matmul(A, B)</p> 
<p>Python version: onnx_ops.matmul(A, B)</p> 

## onnx::GRU
 <p><i>
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p> 
<p><b>onnx::GRU(X, W, R, B, sequence_lens, initial_h)</b> = onnx_ops.gru(X, W, R, B, sequence_lens, initial_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, linear_before_reset)</p> 
<p>Python version: onnx_ops.gru(X, W, R, B, sequence_lens, initial_h, activation_alpha, activation_beta, activations, clip, direction, hidden_size, linear_before_reset)</p> 

## onnx::Resize
 <p><i>
Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.
</i></p> 
<p><b>onnx::Resize(X, roi, scales, sizes)</b> = onnx_ops.resize(X, roi, scales, sizes, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode)</p> 
<p>Python version: onnx_ops.resize(X, roi, scales, sizes, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode)</p> 

## onnx::GlobalLpPool
 <p><i>
 GlobalLpPool consumes an input tensor X and applies lp pool pooling across
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.</i></p> 
<p><b>onnx::GlobalLpPool(X)</b> = onnx_ops.globallppool(X, p)</p> 
<p>Python version: onnx_ops.globallppool(X, p)</p> 

## onnx::SequenceConstruct
 <p><i>
Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.
</i></p> 
<p><b>onnx::SequenceConstruct(inputs)</b> = onnx_ops.sequenceconstruct(inputs)</p> 
<p>Python version: onnx_ops.sequenceconstruct(inputs)</p> 

## onnx::Elu
 <p><i>
Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.

</i></p> 
<p><b>onnx::Elu(X)</b> = onnx_ops.elu(X, alpha)</p> 
<p>Python version: onnx_ops.elu(X, alpha)</p> 

## onnx::GlobalAveragePool
 <p><i>
 GlobalAveragePool consumes an input tensor X and applies average pooling across
 the values in the same channel. This is equivalent to AveragePool with kernel size
 equal to the spatial dimension of input tensor.</i></p> 
<p><b>onnx::GlobalAveragePool(X)</b> = onnx_ops.globalaveragepool(X)</p> 
<p>Python version: onnx_ops.globalaveragepool(X)</p> 

## onnx::Tan
 <p><i>
Calculates the tangent of the given input tensor, element-wise.
</i></p> 
<p><b>onnx::Tan(input)</b> = onnx_ops.tan(input)</p> 
<p>Python version: onnx_ops.tan(input)</p> 

## onnx::Exp
 <p><i>
Calculates the exponential of the given input tensor, element-wise.
</i></p> 
<p><b>onnx::Exp(input)</b> = onnx_ops.exp(input)</p> 
<p>Python version: onnx_ops.math.exp(input)</p> 

## onnx::Unique
 <p><i>
Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
The first output tensor 'Y' contains all unique values or subtensors of the input.
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurance in 'X'..
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'. ".
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
  input_X = [2, 1, 1, 3, 4, 3]
  attribute_sorted = 0
  attribute_axis = None
  output_Y = [2, 1, 3, 4]
  output_indices = [0, 1, 3, 4]
  output_inverse_indices = [0, 1, 1, 2, 3, 2]
  output_counts = [1, 2, 2, 1]

Example 2:
  input_X = [[1, 3], [2, 3]]
  attribute_sorted = 1
  attribute_axis = None
  output_Y = [1, 2, 3]
  output_indices = [0, 2, 1]
  output_inverse_indices = [0, 2, 1, 2]
  output_counts = [1, 1, 2]

Example 3:
  input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
  attribute_sorted = 1
  attribute_axis = 0
  output_Y = [[1, 0, 0], [2, 3, 4]]
  output_indices = [0, 2]
  output_inverse_indices = [0, 0, 1]
  output_counts = [2, 1]

Example 4:
  input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
             [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
  attribute_sorted = 1
  attribute_axis = 1

  intermediate data are presented below for better understanding:

  there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
  A: [[1, 1], [1, 1]],
     [[0, 1], [0, 1]],
     [[2, 1], [2, 1]],
     [[0, 1], [0, 1]].

  there are 3 unique subtensors:
  [[1, 1], [1, 1]],
  [[0, 1], [0, 1]],
  [[2, 1], [2, 1]].

  sorted unique subtensors:
  B: [[0, 1], [0, 1]],
     [[1, 1], [1, 1]],
     [[2, 1], [2, 1]].

  output_Y is constructed from B:
  [[[0. 1.], [1. 1.], [2. 1.]],
   [[0. 1.], [1. 1.], [2. 1.]]]

  output_indices is to map from B to A:
  [1, 0, 2]

  output_inverse_indices is to map from A to B:
  [1, 0, 2, 0]

  output_counts = [2 1 1]
</i></p> 
<p><b>onnx::Unique(X)</b> = onnx_ops.unique(X, axis, sorted)</p> 
<p>Python version: onnx_ops.unique(X, axis, sorted)</p> 

## onnx::ArgMin
 <p><i>
Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulting tensor have the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.</i></p> 
<p><b>onnx::ArgMin(data)</b> = onnx_ops.argmin(data, axis, keepdims, select_last_index)</p> 
<p>Python version: onnx_ops.argmin(data, axis, keepdims, select_last_index)</p> 

## onnx::Add
 <p><i>
Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Add(A, B)</b> = onnx_ops.add(A, B)</p> 
<p>Python version: onnx_ops.add(A, B)</p> 

## onnx::Constant
 <p><i>
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
</i></p> 
<p><b>onnx::Constant()</b> = onnx_ops.constant(sparse_value, value, value_float, value_floats, value_int, value_ints, value_string, value_strings)</p> 
<p>Python version: onnx_ops.constant(sparse_value, value, value_float, value_floats, value_int, value_ints, value_string, value_strings)</p> 

## onnx::Equal
 <p><i>
Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Equal(A, B)</b> = onnx_ops.equal(A, B)</p> 
<p>Python version: onnx_ops.equal(A, B)</p> 

## onnx::ReduceSum
 <p><i>
Computes the sum of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceSum(data, axes)</b> = onnx_ops.reducesum(data, axes, keepdims, noop_with_empty_axes)</p> 
<p>Python version: onnx_ops.reducesum(data, axes, keepdims, noop_with_empty_axes)</p> 

## onnx::Pow
 <p><i>
Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).</i></p> 
<p><b>onnx::Pow(X, Y)</b> = onnx_ops.pow(X, Y)</p> 
<p>Python version: onnx_ops.pow(X, Y)</p> 

## onnx::MaxPool
 <p><i>
 MaxPool consumes an input tensor X and applies max pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 max pooling consisting of computing the max on all values of a
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

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is maximum number of elements exclude pad. 
 </i></p> 
<p><b>onnx::MaxPool(X)</b> = onnx_ops.maxpool(X, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides)</p> 
<p>Python version: onnx_ops.maxpool(X, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides)</p> 

## onnx::Min
 <p><i>
Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Min(data_0)</b> = onnx_ops.min(data_0)</p> 
<p>Python version: onnx_ops.min(data_0)</p> 

## onnx::Div
 <p><i>
Performs element-wise binary division (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Div(A, B)</b> = onnx_ops.div(A, B)</p> 
<p>Python version: onnx_ops.div(A, B)</p> 

## onnx::ReduceMean
 <p><i>
Computes the mean of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.</i></p> 
<p><b>onnx::ReduceMean(data)</b> = onnx_ops.reducemean(data, axes, keepdims)</p> 
<p>Python version: onnx_ops.reducemean(data, axes, keepdims)</p> 

## onnx::Less
 <p><i>
Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
</i></p> 
<p><b>onnx::Less(A, B)</b> = onnx_ops.less(A, B)</p> 
<p>Python version: onnx_ops.less(A, B)</p> 

## onnx::Dropout
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
<p><b>onnx::Dropout(data, ratio, training_mode)</b> = onnx_ops.dropout(data, ratio, training_mode, seed)</p> 
<p>Python version: onnx_ops.dropout(data, ratio, training_mode, seed)</p> 

## onnx::DepthToSpace
 <p><i>DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])

tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])

y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])


In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])

tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])

y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])

</i></p> 
<p><b>onnx::DepthToSpace(input)</b> = onnx_ops.depthtospace(input, blocksize, mode)</p> 
<p>Python version: onnx_ops.depthtospace(input, blocksize, mode)</p> 

## onnx::Ceil
 <p><i>
Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.
</i></p> 
<p><b>onnx::Ceil(X)</b> = onnx_ops.ceil(X)</p> 
<p>Python version: onnx_ops.ceil(X)</p> 

## onnx::Atan
 <p><i>
Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
</i></p> 
<p><b>onnx::Atan(input)</b> = onnx_ops.atan(input)</p> 
<p>Python version: onnx_ops.atan(input)</p> 

## onnx::LogSoftmax
 <p><i>
The operator computes the log of softmax values for the given input:

 LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

The input does not need to explicitly be a 2D vector. The "axis" attribute
indicates the dimension along which LogSoftmax will be performed.
The output tensor has the same shape
and contains the LogSoftmax values of the corresponding input.
</i></p> 
<p><b>onnx::LogSoftmax(input)</b> = onnx_ops.logsoftmax(input, axis)</p> 
<p>Python version: onnx_ops.logsoftmax(input, axis)</p> 

## onnx::AveragePool
 <p><i>
 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 </i></p> 
<p><b>onnx::AveragePool(X)</b> = onnx_ops.averagepool(X, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides)</p> 
<p>Python version: onnx_ops.averagepool(X, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides)</p> 

## onnx::Hardmax
 <p><i>
The operator computes the hardmax values for the given input:

 Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

The input does not need to explicitly be a 2D vector. The "axis" attribute
indicates the dimension along which Hardmax will be performed.
The output tensor has the same shape
and contains the Hardmax values of the corresponding input.
</i></p> 
<p><b>onnx::Hardmax(input)</b> = onnx_ops.hardmax(input, axis)</p> 
<p>Python version: onnx_ops.hardmax(input, axis)</p> 

## onnx::Abs
 <p><i>
Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
</i></p> 
<p><b>onnx::Abs(X)</b> = onnx_ops.abs(X)</p> 
<p>Python version: onnx_ops.abs(X)</p> 

## onnx::ConvTranspose
 <p><i>
The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    </i></p> 
<p><b>onnx::ConvTranspose(X, W, B)</b> = onnx_ops.convtranspose(X, W, B, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides)</p> 
<p>Python version: onnx_ops.convtranspose(X, W, B, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides)</p> 

## onnx::LpNormalization
 <p><i>
Given a matrix, apply Lp-normalization along the provided axis.
</i></p> 
<p><b>onnx::LpNormalization(input)</b> = onnx_ops.lpnormalization(input, axis, p)</p> 
<p>Python version: onnx_ops.lpnormalization(input, axis, p)</p> 

## onnx::Gemm
 <p><i>General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
</i></p> 
<p><b>onnx::Gemm(A, B, C)</b> = onnx_ops.gemm(A, B, C, alpha, beta, transA, transB)</p> 
<p>Python version: onnx_ops.gemm(A, B, C, alpha, beta, transA, transB)</p> 

## change_goal
 <p><i>ACT-R change goal buffer function</i></p> 
<p><b>change_goal(pattern, curr_goal)</b> = actr_functions.change_goal(pattern, curr_goal)</p> 
<p>Python version: actr_functions.change_goal(pattern, curr_goal)</p> 

## retrieve_chunk
 <p><i>ACT-R retrieve chunk function</i></p> 
<p><b>retrieve_chunk(pattern, dm_chunks, types)</b> = actr_functions.retrieve_chunk(pattern, dm_chunks, types)</p> 
<p>Python version: actr_functions.retrieve_chunk(pattern, dm_chunks, types)</p> 

## pattern_matching_function
 <p><i>ACT-R pattern matching function</i></p> 
<p><b>pattern_matching_function(productions, goal, retrieval)</b> = actr_functions.pattern_matching_function(productions, goal, retrieval)</p> 
<p>Python version: actr_functions.pattern_matching_function(productions, goal, retrieval)</p> 

## conflict_resolution_function
 <p><i>ACT-R conflict resolution function</i></p> 
<p><b>conflict_resolution_function(productions)</b> = actr_functions.conflict_resolution_function(productions)</p> 
<p>Python version: actr_functions.conflict_resolution_function(productions)</p> 

## update_goal
 <p><i>ACT-R update goal buffer function</i></p> 
<p><b>update_goal(production)</b> = actr_functions.update_goal(production)</p> 
<p>Python version: actr_functions.update_goal(production)</p> 

## update_retrieval
 <p><i>ACT-R update retrieval buffer function</i></p> 
<p><b>update_retrieval(production)</b> = actr_functions.update_retrieval(production)</p> 
<p>Python version: actr_functions.update_retrieval(production)</p> 

## check_termination
 <p><i>check_termination</i></p> 
<p><b>check_termination(production)</b> = actr_functions.check_termination(production)</p> 
<p>Python version: actr_functions.check_termination(production)</p> 
