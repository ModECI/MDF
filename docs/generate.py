from modeci_mdf.mdf import *

from modeci_mdf import MODECI_MDF_VERSION
import json
import types
import yaml
import shutil

shutil.copy("../README.md", "sphinx/source/api/Introduction.md")

for ex in [
    "ACT-R",
    "MDF",
    "NeuroML",
    "ONNX",
    "PsyNeuLink",
    "PyTorch",
    "Quantum",
    "WebGME",
]:
    shutil.copy(
        "../examples/%s/README.md" % ex,
        f"sphinx/source/api/export_format/{ex}/{ex}.md",
    )


mod = Model(id="Simple")

doc = mod.generate_documentation(format="markdown")

comment = "**Note: the ModECI MDF specification is still in development!** See [here](https://github.com/ModECI/MDF/issues) for ongoing discussions."
comment_rst = "**Note: the ModECI MDF specification is still in development!** See `here <https://github.com/ModECI/MDF/issues>`_ for ongoing discussions."

with open("README.md", "w") as d:
    d.write("# Specification of ModECI v%s\n" % MODECI_MDF_VERSION)
    d.write("%s\n" % comment)
    d.write(doc)
"""
with open("sphinx/source/api/Specification.md", "w") as d:
    d.write("# Specification of ModECI v%s\n" % MODECI_MDF_VERSION)
    d.write("%s\n" % comment)
    d.write(doc)"""


doc = mod.generate_documentation(format="rst")

with open("sphinx/source/api/Specification.rst", "w") as d:
    ver = "Specification of ModECI v%s" % MODECI_MDF_VERSION
    d.write("%s\n" % ("=" * len(ver)))
    d.write("%s\n" % ver)
    d.write("%s\n\n" % ("=" * len(ver)))
    d.write("%s\n\n" % comment_rst)
    d.write(doc)

doc = mod.generate_documentation(format="dict")

doc = {
    "version": "ModECI MDF v%s" % MODECI_MDF_VERSION,
    "comment": comment,
    "specification": doc,
}

with open("MDF_specification.json", "w") as d:
    d.write(json.dumps(doc, indent=4))
with open("MDF_specification.yaml", "w") as d:
    d.write(yaml.dump(doc, indent=4, sort_keys=False))

print("Written main documentation")

# Generate export formats documentation
export_formats_list = [
    "MDF",
    "ACT-R",
    "NeuroML",
    "ONNX",
    "PsyNeuLink",
    "PyTorch",
    "Quantum",
    "WebGME",
]
for format in export_formats_list:
    with open("../examples/" + format + "/README.md") as readfile, open(
        "sphinx/source/api/export_format/" + format + "/" + format + ".md", "w"
    ) as writefile:
        # read content from first file
        for line in readfile:
            # append content to second file
            writefile.write(line)


print("Written Export format documentation")

# Generate standard function generate_documentation

from modeci_mdf.functions.standard import mdf_functions, create_python_expression

mdf_dumpable = {
    name: {
        k: v
        for k, v in mdf_functions[name].items()
        if not isinstance(v, types.FunctionType)
    }
    for name in mdf_functions
}

with open("MDF_function_specifications.json", "w") as d:
    d.write(json.dumps(mdf_dumpable, indent=4))
with open("MDF_function_specifications.yaml", "w") as d:
    d.write(yaml.dump(mdf_dumpable, indent=4, sort_keys=False))


func_doc = ""
with open("sphinx/source/api/MDF_function_specifications.md", "w") as d:
    d.write(
        "# Specification of standard functions in ModECI v%s\n" % MODECI_MDF_VERSION
    )
    d.write("%s\n" % comment)

    d.write(
        "These functions are defined in https://github.com/ModECI/MDF/blob/main/src/modeci_mdf/standard_functions.py\n"
    )

    d.write("## All functions:\n | ")
    all_f = sorted(mdf_functions.keys())
    for f in all_f:
        c = ":"
        n = ""
        d.write(f'<a href="#{f.lower().replace(c,n)}">{f}</a> | ')

    for f in mdf_functions:

        d.write("\n## %s\n " % f)
        func = mdf_functions[f]
        d.write("<p><i>%s</i></p> \n" % (func["description"]))
        # d.write('<p>Arguments: %s</p> \n'%(func['arguments']))

        d.write(
            "<p><b>%s(%s)</b> = %s</p> \n"
            % (f, ", ".join([a for a in func["arguments"]]), func["expression_string"])
        )
        d.write(
            "<p>Python version: %s</p> \n"
            % (create_python_expression(func["expression_string"]))
        )


print("Written function documentation")
