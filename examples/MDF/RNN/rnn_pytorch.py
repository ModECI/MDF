import torch
from torch import nn

input_size = 1
hidden_size = 1
num_layers = 1

in_x = 1
in_y = 1

rnn = nn.RNN(
    input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=False
)

print("RNN: {}; {}".format(rnn, type(rnn)))

print("Model: %s" % rnn)


for i in range(num_layers):
    h = torch.zeros(hidden_size, input_size)
    h[0][0] = 1
    exec("rnn.weight_ih_l%i = torch.nn.Parameter(h)" % i)
    # exec('rnn.weight_ih_l%i[0] = 1'%i)

for l in range(num_layers):
    exec(
        "rnn.weight_hh_l%i = torch.nn.Parameter(torch.zeros(hidden_size,hidden_size))"
        % l
    )
    # exec('rnn.weight_hh_l%i[0][0] = 1'%l)


print("State_dict:")
for s in rnn.state_dict():
    print(" > {} = {}".format(s, rnn.state_dict()[s]))

input = torch.zeros(in_x, in_y, input_size)
input[0][0][0] = 3
print("Input: %s" % input)

h0 = torch.randn(num_layers, in_y, hidden_size)
h0 = torch.zeros(num_layers, in_y, hidden_size)
# h0[0][0]=0.5
print("h0: %s" % h0)

output, hn = rnn(input, h0)

print("\nOutput calculated by pyTorch, output: %s" % output.detach().numpy())
print("hn: %s" % hn.detach().numpy())


"""
print('State_dict:')
for s in rnn.state_dict():
    print(' > %s = %s'%(s,rnn.state_dict()[s]))


# Export the model
fn = "rnn.onnx"
torch_out = torch.onnx._export(rnn,             # model being run
                               input,                       # model input (or a tuple for multiple inputs)
                               fn,       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

print('Done! Exported to: %s'%fn)
"""
