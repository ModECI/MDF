from modeci_mdf.mdf import *

from modeci_mdf import MODECI_MDF_VERSION
from modeci_mdf import __version__
import json
import types
import yaml

mod = Model(id="Simple")

doc = mod.generate_documentation(format="markdown")

comment = "**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**"

with open("README.md", "w") as d:
    d.write("# Specification of ModECI v%s\n" % MODECI_MDF_VERSION)
    d.write("%s\n" % comment)
    d.write(doc)

with open("sphinx/source/api/Specification.md", "w") as d:
    d.write("# Specification of ModECI v%s\n" % MODECI_MDF_VERSION)
    d.write("%s\n" % comment)
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

# Generate standard function generate_documentation

from modeci_mdf.standard_functions import mdf_functions, create_python_expression

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
