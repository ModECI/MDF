
from modeci_mdf.mdf import *

from modeci_mdf import MODECI_MDF_VERSION
from modeci_mdf import __version__

mod = Model(id='Simple')

doc = mod.generate_documentation(format='markdown')
#print(doc)
with open('README.md','w') as d:
    d.write('# Specification of ModECI v%s\n'%MODECI_MDF_VERSION)
    d.write('**Note: specification in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**\n\n')
    d.write(doc)
