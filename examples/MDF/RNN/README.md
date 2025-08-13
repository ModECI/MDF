**These examples are work in progress!**

## Recurrent Neural Network (RNN) model

The script [generate_rnn.py](generate_rnn.py) can be used to generate an example RNN model. The MDF is in [RNNs.yaml](RNNs.yaml) and `python generate_rnn.py -graph` will produce the following graph of the network:

<p align="center"><img src="rnn.png" alt="rnn.png" height="400"></p>

To run the network using the MDF execution engine type `python generate_rnn.py -run`, producing:

<p align="center"><img src="RNN.run.png" alt="RNN.run.png" height="400"></p>


## Integrate and Fire (IaF) neuron model

The script [generate_iaf.py](generate_iaf.py) can be used to generate an example Integrate and Fire model. The MDF is in [IAFs.yaml](IAFs.yaml) and `python generate_iaf.py -graph` will produce the following graph of the network:

<p align="center"><img src="iaf.png" alt="iaf.png"></p>

To run the network with 1 input, a pre and a post IAF node (each with 1 cell) using the MDF execution engine type `python generate_iaf.py -run`, producing:

<p align="center"><img src="IaF.run.png" alt="IaF.run.png" height="500"></p>

To run a network (2 populations, 8 cells each) with an array for the state variable v, type `python generate_iaf.py -run -net`, producing:

<p align="center"><img src="IaF.net.run.png" alt="IaF.net.run.png" height="500"></p>

To run a similar network (2 populations, 8 cells each) with timed pulses into the pre cells, and weighed connections to the post cells, type `python generate_iaf.py -run -net2`, producing:

<p align="center"><img src="IaF.net2.run.png" alt="IaF.net2.run.png" height="500"></p>
