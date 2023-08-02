from modeci_mdf.mdf import *

from modeci_mdf import MODECI_MDF_VERSION
import json
import types
import yaml
import shutil

shutil.copy("../README.md", "sphinx/source/api/Introduction.md")
shutil.copy("../CONTRIBUTING.md", "sphinx/source/api/Contributing.md")

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

import glob

for ex in ["ACT-R", "NeuroML", "ONNX", "PyTorch"]:
    for suf in ["png", "svg"]:
        for file in glob.glob(f"../examples/{ex}/*.{suf}"):
            print("Copying: %s" % file)
            shutil.copy(file, "sphinx/source/api/export_format/%s" % ex)

for file in glob.glob("../examples/MDF/images/*.png"):
    print("Copying: %s" % file)
    shutil.copy(file, "sphinx/source/api/export_format/MDF/images")


mod = Model(id="Simple")
condition = Condition("testing_condition")
condition_set = ConditionSet()

doc = mod.generate_documentation(format="markdown")
doc_md_1 = condition.generate_documentation(format="markdown")
doc_md_2 = condition_set.generate_documentation(format="markdown")

comment = "**Note: the ModECI MDF specification is still in development!** See [here](https://github.com/ModECI/MDF/issues) for ongoing discussions."
comment_rst = "**Note: the ModECI MDF specification is still in development!** See `here <https://github.com/ModECI/MDF/issues>`_ for ongoing discussions."

with open("README.md", "w") as d:
    d.write("# Specification of ModECI v%s\n" % MODECI_MDF_VERSION)
    d.write("%s\n" % comment)
    d.write(doc)
    d.write(doc_md_1)
    d.write(doc_md_2)


doc = mod.generate_documentation(format="rst")
doc_rst_1 = condition.generate_documentation(format="rst")
doc_rst_2 = condition_set.generate_documentation(format="rst")

with open("sphinx/source/api/Specification.rst", "w") as d:
    ver = "Specification of ModECI v%s" % MODECI_MDF_VERSION
    d.write("%s\n" % ("=" * len(ver)))
    d.write("%s\n" % ver)
    d.write("%s\n\n" % ("=" * len(ver)))
    d.write("%s\n\n" % comment_rst)
    d.write(doc)
    d.write(doc_rst_1)
    d.write(doc_rst_2)

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
    for name in sorted(mdf_functions.keys())
}

with open("MDF_function_specifications.json", "w") as d:
    d.write(json.dumps(mdf_dumpable, indent=4))
with open("MDF_function_specifications.yaml", "w") as d:
    d.write(yaml.dump(mdf_dumpable, indent=4, sort_keys=False))


func_doc = ""
with open("MDF_function_specifications.md", "w") as d:
    d.write(
        "# Specification of standard functions in ModECI v%s\n" % MODECI_MDF_VERSION
    )
    d.write("%s\n" % comment)

    d.write(
        "These functions are defined in Python API module "
        '<a href="https://github.com/ModECI/MDF/tree/main/src/modeci_mdf/functions">modeci_mdf.functions</a>.\n'
    )

    all_f = sorted(mdf_functions.keys())
    onnx_f = [f for f in all_f if f.startswith("onnx::")]
    non_onnx_f = [f for f in all_f if not f.startswith("onnx::")]

    d.write("## Non-ONNX Functions\n\n")
    for f in non_onnx_f:
        d.write(f'- <a href="#{f.lower().replace("_", "")}">{f}</a>\n')

    d.write("\n## ONNX Functions\n\n")
    for f in onnx_f:
        f = f.replace("onnx::", "")
        d.write(f'- <a href="#{f.lower().replace("_", "")}">{f}</a>\n')

    for f in sorted(mdf_functions.keys()):
        f_str = f.replace("onnx::", "")
        d.write(f"<a name=\"{f_str.lower().replace('_', '')}\"></a>\n\n")
        d.write("## %s\n " % f_str)
        func = mdf_functions[f]

        d.write("<p><i>%s</i></p> \n" % (func["description"]))
        # d.write('<p>Arguments: %s</p> \n'%(func['arguments']))

        if "onnx::" not in f:
            d.write(
                "<p><b>%s(%s)</b> = %s</p> \n"
                % (
                    f,
                    ", ".join([a for a in func["arguments"]]),
                    func["expression_string"],
                )
            )

        d.write(
            "\nPython version: `%s`\n\n"
            % (create_python_expression(func["expression_string"]))
        )

        if "onnx::" in f:
            # Link to ONNX API Documentation
            onnx_link = f"https://onnx.ai/onnx/operators/onnx__{f_str}.html"
            d.write(f"<a href={onnx_link}><i>ONNX Documentation</i></a>\n")

shutil.copy(
    "MDF_function_specifications.md", "sphinx/source/api/MDF_function_specifications.md"
)

print("Written function documentation")
