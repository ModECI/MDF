# Conditions in MDF

In MDF, Conditions are used to specify how many times and when individual nodes are allowed to be executed. MDF Conditions are created using the Graphscheduler Library and to use conditions in MDF, the graphscheduler library must be installed. There are different types of conditions which are categorized into six and they includes:

**Generic Condition** - This is satisfied when a user-specified function and set of arguments evaluates to **True**, They are also used for calling [custom conditions](https://kmantel.github.io/graph-scheduler/Condition.html#condition-custom). Examples of Conditions that are categorized under Generic condition can be found [here](https://kmantel.github.io/graph-scheduler/Condition.html#conditions-generic). see [here](https://github.com/ModECI/MDF/blob/main/examples/MDF/conditions/threshold.py) for generic condition example in MDF.

**Static Condition** - This condition is satisfied either always or never and they are independent of other Conditions, nodes or time. Example of the use of a static condition in MDF can be found [here](https://github.com/ModECI/MDF/blob/main/examples/MDF/conditions/everyNCalls.py). Conditions that are categorized as static can be found [here](https://kmantel.github.io/graph-scheduler/Condition.html#conditions-static)

**Composite Condition** - This condition is dependent on another condition(s) and it is satisfied based on the condition it is dependent on. Conditions that are categorized as composite can be found [here](https://kmantel.github.io/graph-scheduler/Condition.html#conditions-composite). Example the use of a composite condition in MDF can be found [here](composite_condition_example)

**Time-based Condition** - This condition is satisfied based on the current count of units of time at a specified TimeScale. Conditions that are categorized as time based can be found [here](https://kmantel.github.io/graph-scheduler/Condition.html#conditions-time-based). See time-based condition use in MDF [here](https://github.com/ModECI/MDF/blob/main/examples/MDF/conditions/timeInterval.py)

**Node-based Condition**- This condition is based on the execution or state of other nodes. Conditions that are categorized as node based can be found [here](https://kmantel.github.io/graph-scheduler/Condition.html#conditions-node-based). Example the use of a node based condition in MDF can be found [here](https://github.com/ModECI/MDF/blob/main/examples/MDF/conditions/everyNCalls.py)

**Convenience Condition** - This condition is based on other Conditions, condensed for convenience. Conditions that are categorized as convenience can be found [here](https://kmantel.github.io/graph-scheduler/Condition.html#conditions-convenience)

These different category of conditions can be used interchangeably for different nodes in a graphs (e.g if there are two nodes in a graph Static conditions can be used for the first node and Time based condition can be used for the second node) Likewise, a condition category  can also be used for different nodes in a graph(e.g if there are two nodes in a graph, Node based conditions can be used for the two nodes). To read more about conditions in MDF see [here](https://kmantel.github.io/graph-scheduler/Condition.html)


## Conditions Examples

These are some graphical examples illustrating the different applications of different categories of conditions in MDF.

Examples of Python, JSON and YAML files illustrating the use of conditions in MDF.

[Threshold](#threshold-example) | [Time Interval](#time-interval-example) | [EveryNCalls](#everyncalls-example) | [Composite](#composite-example)
### Generic condition example

[Python source](threshold.py)  | [JSON](threshold_condition.json) | [YAML](threshold_condition.yaml)

A simple example with 1 [Node](../../../docs/README.md#node)

<p align="center"><img src="images/threshold.png" alt="threshold"/></p>

**NOTE:** Looking at the graph above, the threshold condition is used to terminate the model when the comparison between the value of the parameter(param_A) and threshold using the comparator evaluates to **TRUE**. For this condition to be satisfied, the graph node **A** must be run 5 times because the threshold condition is set to be greater or equal to five which will make the order of executions of nodes to be: (A ,A, A, A, A) after which the output of A will be five and the condition will be satisfied and terminated.


###  Static and Node based conditions Examples

[Python source](everyNCalls.py) | [JSON](everyncalls_condition.json) | [YAML](everyncalls_condition.yaml)

Another simple example with 3 [Nodes](../../../docs/README.md#node).

<p align="center"><img width="400" src="images/everyncalls.png" alt="everyncalls"/></p>

**NOTE:** Looking at the graph above, the conditions used in this case are used as a node specific conditions. The first node (A) is not dependent on any node and it will always run in this case, the second node(B) is dependent on the first node as it will only run after the first node(A) has been ran two times, it wont run if this condition is not satisfied, third node(C) is dependent on the second node(B) as it will only run if the B node has ran three times. But because the B node is dependent on A node to run, the C node has to inherit the dependency of the B nodes, i.e the order of execution of this graph will be: ( A, A, B, A, A, B, A, A, B, C). In simpler terms, the A node will always run (A), the B node will run only when the A node has been run two times (A, A, B), The C nodes will only run when the B nodes has run for three times (A, A, B, A, A, B, A, A, B, C)

### Time Interval condition example

[Python source](timeInterval.py) | [JSON](timeinterval_condition.json) | [YAML](timeinterval_condition.yaml)

A simple 2 [Nodes](../../../docs/README.md#node) graph satisfying the [ Time Interval Conditions](https://kmantel.github.io/graph-scheduler/Condition.html#graph_scheduler.condition.TimeInterval)

<p align="center"><img src="images/timeinterval.png" alt="time interval"/></p>

**NOTE:** Looking at the graph above, The time interval condition is used as a node specific condition and each nodes not dependent on each other. The first node (A) will run after five milli-seconds while the second node (B) will run after ten milli-seconds. This condition can make the nodes run indefinitely so the conditional statements (if, while) is strongly advised to be used when evaluating graphs that uses time interval condition. In this case, the condition will remain true until t > duration. The order of execution of this graph is (A, A, A, B, A, B, A, B, A, B, A, B, A, B). The A node ran after 5ms (A), then it ran again after 5ms i.e 5+5= 10 (A, A), then it will run again after 5ms: 5+5+5= 15ms and because the time condition is now satisfying the B node, the B condition will also run (A, A, A, B). After which it will continue to run until the conditional statement is no longer **TRUE** and the trial will be terminated.



### Composite Condition Example

[Python Source](composite_condition_example.py) | [JSON](Composite_mdf_condition.json) | [YAML](Composite_mdf_condition.yaml)

A simple example with 3 [Nodes](../../../docs/README.md#node).

<p align="center"><img width="400" src="images/composite_example.png" alt="composite condition"/></p>

**NOTE:** Looking at the graph above, the **All** condition is used to terminate the execution when the all condition dependencies has been met. The dependency used in this graph is the **aftercall** which is set on the A node and B node. The B node will override the A node because it has the highest **n** numbers (n is the number of executions of dependency after which the Condition is satisfied). The order of execution of nodes will be ( A, B, C, A, B, C, A, B, C, A, B). The A, B and C nodes will run consecutively for three times each because n= 3 in this graph and after which A and B nodes will run once and the trial will be terminated
