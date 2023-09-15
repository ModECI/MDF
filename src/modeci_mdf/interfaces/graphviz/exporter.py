"""
Simple export of MDF to GraphViz for generating graphics.

Work in progress...
"""

import sys

import graphviz
import numpy as np

from modeci_mdf.functions.standard import mdf_functions

from modeci_mdf.utils import color_rgb_to_hex

from modelspec.utils import _val_info

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
COLOR_BG_MAIN = "#ffffff"
COLOR_LABEL = "#666666"
COLOR_NUM = "#444444"
COLOR_PARAM = "#1666ff"
COLOR_INPUT = "#188855"
COLOR_FUNC = "#441199"
COLOR_OUTPUT = "#cc3355"
COLOR_COND = "#ffa1d"
COLOR_TERM = COLOR_COND  # same as conditions


def format_label(s):
    # return f'<font color="{COLOR_LABEL}"><b>{s}</b></font></td><td>'
    return ""


def format_num(s):
    if type(s) == np.ndarray:
        ss = "%s" % (np.array2string(s, threshold=4, edgeitems=1))
        info = f" (NP {s.shape} {s.dtype})"
    else:
        ss = s
        info = ""
    return f'<font color="{COLOR_NUM}"><b>{ss}</b>{info}</font>'


def format_bold(s, use_bold=True):
    return f"<b>{s}</b>" if use_bold else s


def format_param(s):
    return f'<font color="{COLOR_PARAM}">{s}</font>'


def format_input(s):
    return f'<font color="{COLOR_INPUT}">{s}</font>'


def format_function(s):
    return f'<font color="{COLOR_FUNC}">{s}</font>'


def format_standard_func(s):
    return "<i>%s</i>" % (s)


def format_standard_func_long(s):
    return "<i>%s</i>" % (s)


def format_output(s):
    return f'<font color="{COLOR_OUTPUT}">{s}</font>'


def format_condition(s):
    return f'<font color="{COLOR_COND}">{s}</font>'


def format_term_condition(s):
    return f'<font color="{COLOR_TERM}">{s}</font>'


def match_in_expr(expr, node):

    if type(expr) != str:
        return "%s" % _val_info(expr)
    else:
        # print("Checking %s" % (expr))

        expr = " %s " % safe_comparitor(expr)

        def _replace_var(v, expr, format_method):
            # print(f"Replacing {v} in {expr}")
            if expr == v:
                return format_method(expr)
            can_start = [" ", "+", "-", "*", "/", "("]
            can_end = [" ", "+", "-", "*", "/", ")"]
            for s in can_start:
                for e in can_end:
                    expr = expr.replace(s + v + e, s + format_method(v) + e)
            return expr

        for p in node.parameters:
            expr = _replace_var(p.id, expr, format_param)

        for ip in node.input_ports:
            expr = _replace_var(ip.id, expr, format_input)

        for f in node.functions:
            expr = _replace_var(f.id, expr, format_function)

        for op in node.output_ports:
            expr = _replace_var(op.id, expr, format_output)

        # print("Checked %s" % (expr))

        return expr.strip()


def safe_comparitor(comp):
    return comp.replace("<", "&lt;").replace(">", "&gt;")


def mdf_to_graphviz(
    mdf_graph,
    engine="dot",
    output_format="png",
    view_on_render=False,
    level=LEVEL_2,
    filename_root=None,
    is_horizontal=False,
    solid_color=False,
):

    DEFAULT_POP_SHAPE = "ellipse"
    DEFAULT_ARROW_SHAPE = "empty"

    print(
        f"Converting MDF graph: {mdf_graph.id} to graphviz (level: {level}, format: {output_format})"
    )

    graph = graphviz.Digraph(
        mdf_graph.id,
        filename="%s.gv" % mdf_graph.id if not filename_root else filename_root,
        engine=engine,
        format=output_format,
        graph_attr={"rankdir": "LR"}
        if is_horizontal
        else None,  # to make the graph horizontal
    )
    # graph termination condition(s) added globally
    global_term_cond_present = False

    if mdf_graph.conditions is not None and mdf_graph.conditions.termination:
        global_term_cond_present = True
        color = COLOR_MAIN
        penwidth = "2"
        graph.attr(
            "node",
            color=COLOR_TERM,
            style="rounded",
            shape="box",
            fontcolor=COLOR_MAIN,
            penwidth=penwidth,
        )
        info = '<table border="0" cellborder="0">'
        info += '<tr><td colspan="2"><b>%s</b></td></tr>' % (
            "Applies to All Nodes in the Graph"
        )
        nt = mdf_graph.conditions.termination["environment_state_update"]
        args = nt.kwargs
        if nt.type == "Threshold":
            info += (
                "<tr><td>{}{} = Satisfied when <b>{}</b> <b>{}</b> <b>{}</b>".format(
                    format_label(" "),
                    format_term_condition("Termination cond"),
                    args.get("parameter"),
                    safe_comparitor(args.get("comparator")),
                    args.get("threshold"),
                )
            )
            info += "</td></tr>"
        if nt.type == "And":
            info += "<tr><td>{}{} = All conditions in the Graph are satisfied".format(
                format_label(" "),
                format_term_condition("Termination cond"),
            )
            info += "</td></tr>"
        if nt.type == "All":
            info += "<tr><td>{}{} = Satisfied when".format(
                format_label(" "),
                format_term_condition("Termination cond"),
            )
            i = 0
            for item in args.get("dependencies"):
                while i < (len(args.get("dependencies")) - 1):
                    info += " <b>{}</b> condition on node <b>{}</b> has ran after <b>{}</b> times and ".format(
                        item.type, item.kwargs["dependencies"], item.kwargs["n"]
                    )
                    i = i + 1

            info += " <b>{}</b> condition on node <b>{}</b> has ran after <b>{}</b> times. ".format(
                item.type, item.kwargs["dependencies"], item.kwargs["n"]
            )

            info += "</td></tr>"
        info += "</table>"
        graph.node("termination condition", label="<%s>" % info)

    for node in mdf_graph.nodes:
        print("    Node: %s" % node.id)
        color = COLOR_MAIN
        fillcolor = COLOR_BG_MAIN
        penwidth = "1"

        if node.metadata is not None:
            if "color" in node.metadata:
                color = color_rgb_to_hex(node.metadata["color"])
                penwidth = "2"

        if solid_color:
            rgb_ = None
            if node.metadata is not None and "color" in node.metadata:
                fillcolor = color
                rgb_ = node.metadata["color"].split(" ")

                if (
                    float(rgb_[0]) * 0.299
                    + float(rgb_[1]) * 0.587
                    + float(rgb_[2]) * 0.2
                ) > 0.45:
                    fcolor = "black"
                else:
                    fcolor = "white"
            else:
                fcolor = "black"

            # print(f"Bkgd color: {rgb_} ({color}), font: {fcolor}")

            graph.attr(
                "node",
                color=color,
                fillcolor=fillcolor,
                style="rounded,filled",
                shape="box",
                fontcolor=fcolor,
                penwidth=penwidth,
            )
        else:
            graph.attr(
                "node",
                color=color,
                style="rounded",
                shape="box",
                fontcolor=COLOR_MAIN,
                penwidth=penwidth,
            )

        info = '<table border="0" cellborder="0">'
        info += '<tr><td colspan="2"><b>%s</b></td></tr>' % (node.id)

        if node.metadata is not None and level >= LEVEL_3:

            info += "<tr><td>%s" % format_label("METADATA")

            for m in node.metadata:
                info += format_standard_func_long("{} = {}".format(m, node.metadata[m]))

            info += "</td></tr>"

        if level >= LEVEL_2:

            if node.input_ports and len(node.input_ports) > 0:
                for ip in node.input_ports:

                    additional = ""
                    if ip.shape is not None:
                        additional += "shape: %s, " % str(ip.shape)
                    """if ip.reduce is not None:
                        if ip.reduce != "overwrite":  # since this is the default...
                            additional += "reduce: %s, " % str(ip.reduce)"""
                    if ip.type is not None:
                        additional += "type: %s, " % str(ip.type)

                    if len(additional) > 0:
                        additional = "(%s)" % additional[:-2]

                    info += "<tr><td>{}{} {}</td></tr>".format(
                        format_label("IN"),
                        format_input(ip.id),
                        additional,
                    )

            if node.parameters and len(node.parameters) > 0:

                for p in node.parameters:
                    try:
                        stateful = p.is_stateful()
                    except:
                        stateful = False
                    if p.function is not None:
                        argstr = (
                            ", ".join(
                                [match_in_expr(str(p.args[a]), node) for a in p.args]
                            )
                            if p.args
                            else "???"
                        )
                        info += "<tr><td>{}{} = {}({})</td></tr>".format(
                            format_label(" "),
                            format_bold(format_param(p.id), stateful),
                            format_standard_func(p.function),
                            argstr,
                        )
                        if level >= LEVEL_3:
                            func_info = mdf_functions[p.function]
                            info += "<tr><td>%s</td></tr>" % (
                                format_standard_func_long(
                                    "%s(%s) = %s"
                                    % (
                                        p.function,
                                        ", ".join([a for a in p.args]),
                                        func_info["expression_string"],
                                    )
                                )
                            )
                    else:
                        v = ""
                        if p.value is not None:
                            val = match_in_expr(p.value, node)
                            v += val
                        if p.default_initial_value is not None:
                            v += "<i>def init value:</i> %s" % match_in_expr(
                                p.default_initial_value, node
                            )
                        if p.time_derivative is not None:
                            v += ", <i>d/dt:</i> %s" % match_in_expr(
                                p.time_derivative, node
                            )
                        for cond in p.conditions:
                            test = cond.test.replace(">", "&gt;").replace("<", "&lt;")
                            v += "<br/><i>{}: </i>IF {} THEN {}={}".format(
                                cond.id,
                                match_in_expr(test, node),
                                format_param(p.id),
                                match_in_expr(cond.value, node),
                            )
                        info += "<tr><td>{}{} = {}</td></tr>".format(
                            format_label(" "),
                            format_bold(format_param(p.id), stateful),
                            v,
                        )

            if node.functions and len(node.functions) > 0:
                for f in node.functions:
                    if f.function is not None:
                        argstr = (
                            ", ".join(
                                [match_in_expr(str(f.args[a]), node) for a in f.args]
                            )
                            if f.args
                            else "???"
                        )
                        info += "<tr><td>{}{} = {}({})</td></tr>".format(
                            format_label("FUNC"),
                            format_function(f.id),
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
                    elif f.value is not None:
                        argstr = "("
                        if f.args:
                            for a in f.args:
                                argstr += "{}={}, ".format(a, f.args[a])
                        else:
                            argstr += " - no args -  "
                        argstr = argstr[:-2] + ")"

                        info += "<tr><td>{}{} = {} {}</td></tr>".format(
                            format_label("FUNC"),
                            format_function(f.id),
                            format_standard_func(f.value),
                            argstr,
                        )

            # node specific conditions

            if mdf_graph.conditions and mdf_graph.conditions.node_specific != None:
                ns = mdf_graph.conditions.node_specific[node.id]
                args = ns.kwargs
                if ns.type == "EveryNCalls":
                    info += "<tr><td>{}{} = <b>{}</b> will run every <b>{}</b> calls of <b>{}</b>".format(
                        format_label(" "),
                        format_condition("condition"),
                        node.id,
                        args.get("n"),
                        args.get("dependencies"),
                    )
                    info += "</td></tr>"
                elif ns.type == "AfterNCalls":
                    info += "<tr><td>{}{} = <b>{}</b> will run when or after <b>{}</b> calls of <b>{}</b>".format(
                        format_label(" "),
                        format_condition("condition"),
                        node.id,
                        args.get("n"),
                        args.get("dependencies"),
                    )
                    info += "</td></tr>"
                elif ns.type == "AfterCall":
                    info += "<tr><td>{}{} = <b>{}</b> will run after <b>{}</b> calls of <b>{}</b>".format(
                        format_label(" "),
                        format_condition("condition"),
                        node.id,
                        args.get("n"),
                        args.get("dependencies"),
                    )
                    info += "</td></tr>"
                elif ns.type == "TimeInterval":
                    info += (
                        "<tr><td>{}{} = <b>{}</b> will run after <b>{}</b> ms".format(
                            format_label(" "),
                            format_condition("condition"),
                            node.id,
                            args.get("start"),
                        )
                    )
                    info += "</td></tr>"
                elif ns.type == "AfterPass":
                    info += "<tr><td>{}{} = <b>{}</b> will run after <b>{}</b> passes".format(
                        format_label(" "),
                        format_condition("condition"),
                        node.id,
                        args.get("n"),
                    )
                    info += "</td></tr>"
                elif ns.type == "Threshold":
                    info += "<tr><td>{}{} = <b>{}</b> satisfied <b>{}</b> <b>{}</b> <b>{}</b>".format(
                        format_label(" "),
                        format_condition("condition"),
                        ns.type,
                        args.get("parameter"),
                        safe_comparitor(args.get("comparator")),
                        args.get("threshold"),
                    )
                    info += "</td></tr>"
                else:
                    info += "<tr><td>{}{} = <b>{}</b> will <b>{}</b> run".format(
                        format_label(" "),
                        format_condition("condition"),
                        node.id,
                        ns.type,
                    )
                    info += "</td></tr>"

            if node.output_ports and len(node.output_ports) > 0:
                for op in node.output_ports:
                    info += "<tr><td>{}{} = {} {}</td></tr>".format(
                        format_label("OUT"),
                        format_output(op.id),
                        match_in_expr(op.value, node),
                        "(shape: %s)" % str(op.shape)
                        if op.shape is not None
                        else ""
                        if level >= LEVEL_2 and op.shape is not None
                        else "",
                    )

        info += "</table>"

        graph.node(node.id, label="<%s>" % info)

        if global_term_cond_present:
            graph.edge("termination condition", node.id, style="invis")

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
        name = graph.render()
        print("Written graph image to: %s" % name)


if __name__ == "__main__":

    from modeci_mdf.utils import load_mdf, print_summary

    verbose = True

    if len(sys.argv) < 3:
        print(
            "Usage:\n\n  python exporter.py MDF_JSON_FILE level [%s]\n\n" % NO_VIEW
            + "where level = 1, 2 or 3. Include %s to supress viewing generated graph on render\n"
            % NO_VIEW
        )
        exit()

    example = sys.argv[1]
    view = NO_VIEW not in sys.argv

    is_horizontal = "-horizontal" in sys.argv

    model = load_mdf(example)

    mod_graph = model.graphs[0]

    print("------------------")
    print("Loaded Graph:")
    print_summary(mod_graph)

    print("------------------")
    # nmllite_file = example.replace('.json','.nmllite.json')
    mdf_to_graphviz(
        mod_graph,
        engine=engines["d"],
        view_on_render=view,
        level=int(sys.argv[2]),
        is_horizontal=is_horizontal,
        solid_color=False,
    )
