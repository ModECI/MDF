import os
import sys

try:
    import psyneulink.core.scheduling as scheduling
except ImportError as e:
    raise ImportError(
        "Conditional scheduling currently requires psyneulink (pip install psyneulink)"
    ) from e

from modeci_mdf.simple_scheduler import FORMAT_DEFAULT, EvaluableGraph


class EvaluableGraphWithConditions(EvaluableGraph):
    def __init__(self, graph, verbose=False):
        super().__init__(graph, verbose=verbose)

        if self.verbose:
            print("\n  Init scheduler")

        self.scheduler = scheduling.Scheduler(
            graph=self.graph.dependency_dict,
            conditions={
                self.graph.get_node(node): self.parse_condition(cond)
                for node, cond in self.graph.conditions["node_specific"].items()
            },
            termination_conds={
                scale: self.parse_condition(cond)
                for scale, cond in self.graph.conditions["termination"].items()
            },
        )

    def evaluate(self, array_format=FORMAT_DEFAULT):
        print(
            "\nEvaluating graph: %s, root nodes: %s, with array format %s,"
            % (self.graph.id, self.root_nodes, array_format)
        )
        str_conds_nb = "\n  ".join(
            [
                "%s: %s" % (node.id, cond)
                for node, cond in self.scheduler.conditions.conditions.items()
            ]
        )
        str_conds_term = "\n  ".join(
            [
                "%s: %s" % (scale, cond)
                for scale, cond in self.scheduler.termination_conds.items()
            ]
        )
        print(" node-based conditions\n  %s" % str_conds_nb)
        print(" termination conditions\n  %s" % str_conds_term)

        incoming_edges = {n: set() for n in self.graph.nodes}
        for edge in self.graph.edges:
            incoming_edges[self.graph.get_node(edge.receiver)].add(edge)

        for ts in self.scheduler.run():
            if self.verbose:
                print(
                    " Evaluating time step: %s"
                    % self.scheduler.get_clock(None).simple_time
                )
            for node in ts:
                for edge in incoming_edges[node]:
                    self.evaluate_edge(edge, array_format=array_format)
                self.enodes[node.id].evaluate_next(array_format=array_format)

        if self.verbose:
            print("Trial terminated")

    def parse_condition(self, condition):
        try:
            typ = getattr(scheduling.condition, condition["type"])
        except AttributeError as e:
            raise ValueError(
                "Unsupported condition type: %s" % condition["type"]
            ) from e
        except TypeError as e:
            raise TypeError("Invalid condition dictionary: %s" % condition) from e

        for k, v in condition["args"].items():
            new_v = self.graph.get_node(v)
            if new_v is not None:
                # arg is a node id
                condition["args"][k] = new_v

            try:
                if isinstance(v, list):
                    # arg is a list all of conditions
                    new_v = [self.parse_condition(item) for item in v]
                else:
                    # arg is another condition
                    new_v = self.parse_condition(v)
            except (TypeError, ValueError):
                pass
            else:
                condition["args"][k] = new_v

        return typ(**condition["args"])


def main(example_file, verbose=True):

    from modeci_mdf.utils import load_mdf, print_summary

    mod_graph = load_mdf(example_file).graphs[0]

    if verbose:
        print("Loaded Graph:")
        print_summary(mod_graph)

        print("------------------")
    eg = EvaluableGraphWithConditions(mod_graph, verbose)
    eg.evaluate()


if __name__ == "__main__":
    example_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "examples/abc_conditions.json"
    )
    verbose = True
    if len(sys.argv) == 2:
        example_file = sys.argv[1]
        verbose = False

    main(example_file, verbose)
