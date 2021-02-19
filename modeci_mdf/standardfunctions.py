
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

    for func in ['exp','sin']:
        expression_string = expression_string.replace('%s('%func,'math.%s('%func)
    return expression_string

def substitute_args(expression_string, args):
    # TODO, better checks for string replacement
    for arg in args:
        expression_string = expression_string.replace(arg, str(args[arg]))
    return expression_string

# Populate the list of known functions

if len(mdf_functions)==0:

    STANDARD_ARG_0 = 'variable0'

    _add_mdf_function('linear',
                      description='Linear function...',
                      arguments=[STANDARD_ARG_0, 'slope', 'intercept'],
                      expression_string='(%s * slope + intercept)'%(STANDARD_ARG_0))

    _add_mdf_function('logistic',
                      description='Logistic function...',
                      arguments=[STANDARD_ARG_0, 'gain', 'bias', 'offset'],
                      expression_string='1/(1 + exp(-1*gain*(%s + bias) + offset))'%(STANDARD_ARG_0))

    _add_mdf_function('exponential',
                      description='Exp function...',
                      arguments=[STANDARD_ARG_0, 'scale', 'rate', 'bias', 'offset'],
                      expression_string='scale * exp((rate * %s) + bias) + offset'%(STANDARD_ARG_0))

    _add_mdf_function('sin',
                      description='Sine function...',
                      arguments=[STANDARD_ARG_0, 'scale'],
                      expression_string='scale * sin(%s)'%(STANDARD_ARG_0))



if __name__ == "__main__":

    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(mdf_functions)
