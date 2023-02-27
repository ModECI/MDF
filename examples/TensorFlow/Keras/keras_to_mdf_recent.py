#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install graphviz
#conda install python-graphviz
#sudo apt-get install graphviz


# In[2]:


import numpy as np
from tensorflow.keras.models import load_model
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import simple_connect
from modeci_mdf.mdf import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten


# In[3]:


new_model = tf.keras.models.load_model("kr_N_model.h5")
for i in new_model.layers:
    print(i.name)


# In[4]:


# View Weights of the Model
new_model.get_weights()


# In[5]:


weights = new_model.layers[1].get_weights()[1]
bias = new_model.layers[0].get_weights()


# In[6]:



# Create a model containing one graph
mod = Model(id="keras_2_MDF")
mod_graph = Graph(id="Keras_to_MDF_example")

#append the Graph object to the Model
mod.graphs.append(mod_graph)


# In[7]:


dummy_input = np.zeros((1))
input_node = Node(id="Dense")
input_node.parameters.append(Parameter(id="input", value=dummy_input))
input_node.parameters.append(Parameter(id="weight", value=weights[2]))
input_node.parameters.append(Parameter(id="bias", value=bias))

# Add an output port
input_node.output_ports.append(OutputPort(id="output", value = "level"))

# Add node to the graph
mod_graph.nodes.append(input_node)


# In[8]:


# Print current structure as YAML
print(mod_graph.to_yaml())


# In[9]:


#dummy_input = np.zeros((1))
Dense_1 = Node(id="Dense_1")
Dense_1.parameters.append(Parameter(id="input", value=dummy_input))
Dense_1.parameters.append(Parameter(id="weight", value=weights[3]))
Dense_1.parameters.append(Parameter(id="bias", value=bias))

# Add an output port
Dense_1.output_ports.append(OutputPort(id="output", value = "level"))

# Add node to the graph
mod_graph.nodes.append(Dense_1)


# In[10]:


# Print current structure as YAML
print(mod_graph.to_yaml())


# In[11]:


#dummy_input = np.zeros((1))
Dense_1 = Node(id="Dense_1")
Dense_1.parameters.append(Parameter(id="input", value=dummy_input))
Dense_1.parameters.append(Parameter(id="weight", value=weights[3]))
Dense_1.parameters.append(Parameter(id="bias", value=bias))

# Add an output port
Dense_1.output_ports.append(OutputPort(id="output", value = "level"))

# Add node to the graph
mod_graph.nodes.append(Dense_1)


# In[12]:


# Print current structure as YAML
print(mod_graph.to_yaml())


# In[13]:


#dummy_input = np.zeros((1))
Dense_2 = Node(id="Dense_2")
Dense_2.parameters.append(Parameter(id="input", value=dummy_input))
Dense_2.parameters.append(Parameter(id="weight", value=weights[4]))
Dense_2.parameters.append(Parameter(id="bias", value=bias))

# Add an output port
Dense_1.output_ports.append(OutputPort(id="output", value = "level"))

# Add node to the graph
mod_graph.nodes.append(Dense_2)


# In[14]:


# Print current structure as YAML
print(mod_graph.to_yaml())


# In[15]:


#  Save the model to file
mod.to_json_file("keras_to_MDF.json")
mod.to_yaml_file("keras_to_MDF.yaml")


# In[16]:


mod.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=3,
        filename_root="keras_to_MDF",
        is_horizontal=True
    )

from IPython.display import Image
Image(filename="Keras_to_MDF_Example.png")


# In[ ]:





# In[ ]:





# In[ ]:




