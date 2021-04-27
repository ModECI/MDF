"""
Simple export of MDF to GraphViz for generating graphics

Work in progress...

"""

import sys

import graphviz
import numpy as np

from modeci_mdf.standard_functions import mdf_functions

engines = {
    "d": "dot",
    "c": "circo",
    "n": "neato",
    "t": "twopi",
    "f": "fdp",
    "s": "sfdp",
    "p": "patchwork",
}

NO_VIEW = "-noview"

LEVEL_1 = 1
LEVEL_2 = 2
LEVEL_3 = 3

COLOR_MAIN = "#444444"
COLOR_LABEL = "#666666"
COLOR_NUM = "#444444"
COLOR_PARAM = "#999944"
COLOR_INPUT = "#188855"
COLOR_FUNC = "#111199"
COLOR_STATE = "#1666ff"
COLOR_OUTPUT = "#cc3355"


def format_label(s):
    return f'<font color="{COLOR_LABEL}"><b>{s}: </b></font></td><td>'


def format_num(s):
    if type(s) == np.ndarray:
        ss = "%s" % (np.array2string(s, threshold=4, edgeitems=1))
        info = f" (NP {s.shape} {s.dtype})"
    else:
        ss = s
        info = ""
    return f'<font color="{COLOR_NUM}"><b>{ss}</b>{info}</font>'


def format_param(s):
    return f'<font color="{COLOR_PARAM}">{s}</font>'


def format_input(s):
    return f'<font color="{COLOR_INPUT}">{s}</font>'


def format_func(s):
    return f'<font color="{COLOR_FUNC}">{s}</font>'


def format_state(s):
    return '<font color="%s">%s</font>' % (COLOR_STATE, s)


def format_standard_func(s):
    return "<i>%s</i>" % (s)


def format_standard_func_long(s):
    return '<i>%s</i>' % (s)


def format_output(s):
    return f'<font color="{COLOR_OUTPUT}">{s}</font>'


def match_in_expr(s, node):

    if node.parameters:
        for p in node.parameters:
            if p in s:
                s = s.replace(p, format_param(p))

    for ip in node.input_ports:
        if ip.id in s:
            s = s.replace(ip.id, format_input(ip.id))

    for f in node.functions:
        if f.id in s:
            s = s.replace(f.id, format_func(f.id))

    for st in node.states:
        if st.id in s:
            s = s.replace(st.id, format_state(st.id))

    for op in node.output_ports:
        if op.id in s:
            s = s.replace(op.id, format_output(op.id))
    return s


def mdf_to_graphviz(
    mdf_graph, engine="dot", output_format="png", view_on_render=False, level=LEVEL_2
):

    DEFAULT_POP_SHAPE = "ellipse"
    DEFAULT_ARROW_SHAPE = "empty"

    print("Converting MDF graph: %s to graphviz" % (mdf_graph.id))

    graph = graphviz.Digraph(
        mdf_graph.id,
        filename="%s.gv" % mdf_graph.id,
        engine=engine,
        format=output_format,
    )

    for node in mdf_graph.nodes:
        print("    Node: %s" % node.id)
        graph.attr(
            "node", color=COLOR_MAIN, style="rounded", shape="box", fontcolor=COLOR_MAIN
        )
        info = '<table border="0" cellborder="0">'
        info += '<tr><td colspan="2"><b>%s</b></td></tr>' % (node.id)

        if level >= LEVEL_2:
            if node.parameters and len(node.parameters) > 0:

                info += "<tr><td>%s" % format_label("PARAMS")
                for p in node.parameters:
                    nn = format_num(node.parameters[p])
                    info += "{} = {}{}".format(
                        format_param(p),
                        nn,
                        "<br/>" if len(nn) > 40 else ";    ",
                    )
                info = info[:-5]
                info += "</td></tr>"

            if node.input_ports and len(node.input_ports) > 0:
                for ip in node.input_ports:
                    info += "<tr><td>{}{} {}</td></tr>".format(
                        format_label("IN"),
                        format_input(ip.id),
                        "(shape: %s)" % ip.shape
                        if level >= LEVEL_2 and ip.shape is not None
                        else "",
                    )

            if node.functions and len(node.functions) > 0:
                for f in node.functions:
                    argstr = (
                        ", ".join([match_in_expr(str(f.args[a]), node) for a in f.args])
                        if f.args
                        else "???"
                    )
                    info += "<tr><td>{}{} = {}({})</td></tr>".format(
                        format_label("FUNC"),
                        format_func(f.id),
                        format_standard_func(f.function),
                        argstr,
                    )
                    if level >= LEVEL_3:
                        func_info = mdf_functions[f.function]
                        info += '<tr><td colspan="2">%s</td></tr>' % (
                            format_standard_func_long(
                                "%s(%s) = %s"
                                % (
                                    f.function,
                                    ", ".join([a for a in f.args]),
                                    func_info["expression_string"],
                                )
                            )
                        )
                        # info += '<tr><td>%s</td></tr>'%(format_standard_func(func_info['description']))


            if node.states and len(node.states) > 0:
                for st in node.states:
                    v = ''
                    if st.value: v+='<i>value:</i> %s'%match_in_expr(st.value, node)
                    if st.default_initial_value: v+='<i>def init value:</i> %s'%match_in_expr(st.default_initial_value, node)
                    if st.time_derivative: v+=', <i>d/dt:</i> %s'%match_in_expr(st.time_derivative, node)
                    info += "<tr><td>%s%s: %s</td></tr>" % (
                        format_label("STATE"),
                        format_state(st.id),
                        v
                    )

            if node.output_ports and len(node.output_ports) > 0:
                for op in node.output_ports:
                    info += "<tr><td>{}{} = {} {}</td></tr>".format(
                        format_label("OUT"),
                        format_output(op.id),
                        match_in_expr(op.value, node),
                        "(shape: %s)" % op.shape
                        if level >= LEVEL_2 and op.shape is not None
                        else "",
                    )

        info += "</table>"

        graph.node(node.id, label="<%s>" % info)

    for edge in mdf_graph.edges:
        print(f"    Edge: {edge.id} connects {edge.sender} to {edge.receiver}")

        label = "%s" % edge.id
        if level >= LEVEL_2:
            label += " ({} -&gt; {})".format(
                format_output(edge.sender_port),
                format_input(edge.receiver_port),
            )
            if edge.parameters:
                for p in edge.parameters:
                    label += "<br/>{}: <b>{}</b>".format(
                        p, format_num(edge.parameters[p])
                    )

        graph.edge(
            edge.sender,
            edge.receiver,
            arrowhead=DEFAULT_ARROW_SHAPE,
            label="<%s>" % label if level >= LEVEL_2 else "",
        )

    if view_on_render:
        graph.view()
    else:
        graph.render()


if __name__ == "__main__":

    from modeci_mdf.utils import load_mdf, print_summary

    verbose = True

    if len(sys.argv) < 3:
        print(
            "Usage:\n\n  python graphviz.py MDF_JSON_FILE level [%s]\n\n" % NO_VIEW
            + "where level = 1, 2 or 3. Include %s to supress viewing generated graph on render\n"
            % NO_VIEW
        )
        exit()

    example = sys.argv[1]
    view = NO_VIEW not in sys.argv

    model = load_mdf(example)

    mod_graph = model.graphs[0]

    print("------------------")
    print("Loaded Graph:")
    print_summary(mod_graph)

    print("------------------")
    # nmllite_file = example.replace('.json','.nmllite.json')
    mdf_to_graphviz(
        mod_graph, engine=engines["d"], view_on_render=view, level=int(sys.argv[2])
    )
