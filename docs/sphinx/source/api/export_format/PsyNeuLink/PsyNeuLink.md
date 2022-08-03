# Interactions between PsyNeuLink and MDF

## Simple

### ABCD

[Python source](model_ABCD.py) | [JSON](model_ABCD.json) | [Reconstructed source](model_ABCD.reconstructed.py)

An example with four [Nodes](../../docs/README.md#node), as in other environments.

### SimpleLinear

#### SimpleLinear-conditional

[Python source](SimpleLinear-conditional.py) | [JSON](SimpleLinear-conditional.json) | [Reconstructed source](SimpleLinear-conditional.reconstructed.py)

A three-[Node](../../docs/README.md#node) example with [Conditions](../../docs/README.md#condition).

#### SimpleLinear-timing

[Python source](SimpleLinear-timing.py) | [JSON](SimpleLinear-timing.json) | [Reconstructed source](SimpleLinear-timing.reconstructed.py)

The same model as in [SimpleLinear-conditional](#SimpleLinear-conditional) with [Conditions](../../docs/README.md#condition) for timeline scheduling. Note: these conditions are still not fully implemented by the scheduler.

## Nested

### Nested without scheduling

[Python source](model_with_nested_graph.py) | [JSON](model_with_nested_graph.json) | [Reconstructed source](model_with_nested_graph.reconstructed.py)

A model with several [Nodes](../../docs/README.md#node) in two [Graphs](../../docs/README.md#graphs), one of which contains the other.

### Nested with scheduling

[Python source](model_nested_comp_with_scheduler.py) | [JSON](model_nested_comp_with_scheduler.json) | [Reconstructed source](model_nested_comp_with_scheduler.reconstructed.py)

A similar model as in [Nested without scheduling](#Nested-without-scheduling) with [Conditions](../../docs/README.md#condition).

## SimpleFN

[Python source](SimpleFN.py) | [JSON](SimpleFN.json) | [Reconstructed source](SimpleFN.reconstructed.py)

An example with a single [Node](../../docs/README.md#node) using the PsyNeuLink implementation of the [FitzHugh–Nagumo model](https://wikipedia.org/wiki/FitzHugh–Nagumo_model).

### SimpleFN-timing

[Python source](SimpleFN-timing.py) | [JSON](SimpleFN-timing.json) | [Reconstructed source](SimpleFN-timing.reconstructed.py)

The same model as in [SimpleFN](#SimpleFN) with [Conditions](../../docs/README.md#condition) for timeline scheduling.
Note: these conditions are still not fully implemented by the scheduler.

### SimpleFN-conditional

[Python source](SimpleFN-conditional.py) | [JSON](SimpleFN-conditional.json) | [Reconstructed source](SimpleFN-conditional.reconstructed.py)

The same model in [SimpleFN](#SimpleFN) with scheduling [Conditions](../../docs/README.md#condition) that mimic the behavior in [SimpleFN-timing](#SimpleFN-timing).

## Stroop

[Python source](stroop_conflict_monitoring.py) | [JSON](stroop_conflict_monitoring.json) | [Reconstructed source](stroop_conflict_monitoring.reconstructed.py)

A model representing the [Stroop effect](https://en.wikipedia.org/wiki/Stroop_effect) with conflict monitoring that uses [Conditions](../../docs/README.md#condition).
