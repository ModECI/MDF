
mdf_functions = {}


def _add_mdf_function(name, 
                      description,
                      expression_string):

    mdf_functions[name] = {}
    
    mdf_functions[name]['description'] = description
    mdf_functions[name]['expression_string'] = expression_string

# Populate the list of known functions

if len(mdf_functions)==0:

    STANDARD_ARG_0 = 'variable0'

    _add_mdf_function('linear', 
                      description='Linear function...',
                      expression_string='%s*slope'%(STANDARD_ARG_0))

    _add_mdf_function('logistic', 
                      description='Logistic function...',
                      expression_string='1/(1 + math.exp(-1*gain * (%s)))'%(STANDARD_ARG_0))


if __name__ == "__main__":

    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(mdf_functions)