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

The python script used for this example can be executed using `python threshold.py -run`
**NOTE:** Looking at the graph above, the threshold condition is used to terminate the model when the comparison between the value of the parameter(param_A) and threshold using the comparator evaluates to **TRUE**. For this condition to be satisfied, the graph node **A** must be run 5 times because the threshold condition is set to be greater or equal to five which will make the order of executions of nodes to be: (A ,A, A, A, A) after which the output of A will be five and the condition will be satisfied and terminated.


###  Static and Node based conditions Examples

[Python source](everyNCalls.py) | [JSON](everyncalls_condition.json) | [YAML](everyncalls_condition.yaml)

Another simple example with 3 [Nodes](../../../docs/README.md#node).

<p align="center"><img width="400" src="images/everyncalls.png" alt="everyncalls"/></p>

The python script used for this example can be executed using `python everyNCalls.py -run`
**NOTE:** Looking at the graph above, the conditions used in this case are used as a node specific conditions. The first node (A) is not dependent on any node and the condition will always run, the second node(B) is dependent on the first node as it will only run after the first node(A) has been ran two times, it wont run if this condition is not satisfied, third node(C) is dependent on the second node(B) as it will only run if the B node has ran three times. But because the B node is dependent on A node to run, the C node has to inherit the dependency of the B nodes, i.e the order of execution of this graph will be: ( A, A, B, A, A, B, A, A, B, C). In simpler terms, the A node will always run (A), the B node will run only when the A node has been run two times (A, A, B), The C nodes will only run when the B nodes has run for three times (A, A, B, A, A, B, A, A, B, C)

### Time Interval condition example

[Python source](timeInterval.py) | [JSON](timeinterval_condition.json) | [YAML](timeinterval_condition.yaml)

A simple 2 [Nodes](../../../docs/README.md#node) graph satisfying the [ Time Interval Conditions](https://kmantel.github.io/graph-scheduler/Condition.html#graph_scheduler.condition.TimeInterval)

<p align="center"><img src="images/timeinterval.png" alt="time interval"/></p>

The python script used for this example can be executed using `python timeInterval.py -run`
**NOTE:** Looking at the graph above, the time interval condition is used as a node specific condition and each nodes are not dependent on each other. Two conditions are to be satisfied for the time interval condition and they includes the time step and the time interval. The second node (B) condition is to run after five milli-seconds(ms) while the third node (C) will run after ten ms and all nodes must be executed in each time step. This condition can make the nodes run indefinitely so the conditional statements (if, while) is strongly advised to be used when evaluating graphs that uses time interval condition. In this case, the conditional statement used is for the nodes to run when the t(time step) <= duration. This  will remain true until t > duration after which the condition will be terminated. Because this graph is schedule to run while t <= duration (in this case duration = 5), the graph will run in six time steps (remember, numbers that are less or equal to 5 includes:0,1,2,3,4,5) and all nodes must be run in each time step, if the condition of a node has not been met in a time step, the time step will rerun until the condition is met after which the time step will be terminated. In the first time step i.e when t=0, the A node ran at 6ms, the B node ran at 7ms and because the condition of the C node has not been met, the A node ran again after which the B node ran at 10ms, this made the condition of the C node to be satisfied as it will run after 10ms and it ran after which the first t i.e t=0 was terminated because all the nodes has ran in this time step and the order of execution of the graph at t=0 is (A,B,A,B,C) then the second time stamp t=1 was activated. Because the nodes time condition has been satisfied in the first time step, the all the nodes were able to run at a go and the order of execution for t=1 was (A,B,A,B,A,B,C,A,B,C). This continued until the final time step at t=5 where the order of execution of the graph is (A, B, A, B, C, A, B, C, A, B, C, A, B, C, A, B, C, A, B, C). After which it stopped running because the conditional statement is no longer **TRUE** and the trial was be terminated.



### Composite Condition Example

[Python Source](composite_condition_example.py) | [JSON](Composite_mdf_condition.json) | [YAML](Composite_mdf_condition.yaml)

A simple example with 3 [Nodes](../../../docs/README.md#node).

<p align="center"><img width="400" src="images/composite_example.png" alt="composite condition"/></p>


The python script used for this example can be executed using `python composite_condition_example.py -run`
**NOTE:** Looking at the graph above, the **All** condition is used to terminate the execution when all of the condition dependencies has been met. The dependency used in this graph is the **aftercall** (Python script can be found [here](composite_condition_example.py)) which is set on the B node and C node. The AfterCall condition is satisfied when the n (the number of executions of dependency) has been surpassed e.g, if n=2, and the dependency is on B node, the aftercall condition will be met when the B node has been run more than 2 times. The order of execution of this graph is (A, B, C, A, B, C, A, B, C, A, B, C). The B node has an aftercall dependency of n=2 which made the B nodes to run until it has surpass the number of execution set to it and the order of execution for the B node will be (A,B,C,A,B,C,A,B). Because the B nodes has run for more than two times, the condition has been met and the trial was terminated. The C node has an aftercall dependency of n=3 which will made the C node to run until it has surpass the number of execution and the order of execution of the C node will be (A,B,C,A,B,C,A,B,C,A,B,C). When the C node ran for more than Three times, the condition was met and the trial was terminated.
