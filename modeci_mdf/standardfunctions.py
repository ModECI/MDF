
mdf_functions = {}


def _add_mdf_function(name, 
                      description,
                      arguments,
                      expression_string):

    mdf_functions[name] = {}
    
    mdf_functions[name]['description'] = description
    mdf_functions[name]['arguments'] = arguments
    mdf_functions[name]['expression_string'] = expression_string
    
def create_python_expression(expression_string):
    expr = expression_string.replace('exp(','math.exp(')
    return expr

def substitute_args(expression_string, args):
    # TODO, better checks for string replacement
    for arg in args:
        expression_string = expression_string.replace(arg, args[arg])
    return expression_string

# Populate the list of known functions

if len(mdf_functions)==0:

    STANDARD_ARG_0 = 'variable0'

    _add_mdf_function('linear', 
                      description='Linear function...',
                      arguments=[STANDARD_ARG_0,'slope'],
                      expression_string='%s*slope'%(STANDARD_ARG_0))

    _add_mdf_function('logistic', 
                      description='Logistic function...',
                      arguments=[STANDARD_ARG_0,'gain'],
                      expression_string='1/(1 + exp(-1*gain * (%s)))'%(STANDARD_ARG_0))


if __name__ == "__main__":

    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(mdf_functions)