
from modeci_mdf.mdf import *

from modeci_mdf import MODECI_MDF_VERSION
from modeci_mdf import __version__
import json
import yaml

mod = Model(id='Simple')

doc = mod.generate_documentation(format='markdown')

comment = '**Note: specification in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**\n\n'
with open('README.md','w') as d:
    d.write('# Specification of ModECI v%s\n'%MODECI_MDF_VERSION)
    d.write('%s\n'%comment)
    d.write(doc)

from collections import OrderedDict
doc = OrderedDict(mod.generate_documentation(format='dict'))


doc = {'version':'ModECI MDF v%s'%MODECI_MDF_VERSION,
       'comment': comment,
       'specification':doc}

with open('MDF_specification.json','w') as d:
    d.write(json.dumps(doc,indent=4))
with open('MDF_specification.yaml','w') as d:
    d.write(yaml.dump(doc,indent=4,sort_keys=False))
