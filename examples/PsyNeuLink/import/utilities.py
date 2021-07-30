
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import IntegratorFunction
from psyneulink.core.globals.keywords import \
    ACCUMULATOR_INTEGRATOR_FUNCTION, ADAPTIVE_INTEGRATOR_FUNCTION, ADDITIVE_PARAM, \
    DECAY, DEFAULT_VARIABLE, DRIFT_DIFFUSION_INTEGRATOR_FUNCTION, DRIFT_ON_A_SPHERE_INTEGRATOR_FUNCTION, \
    DUAL_ADAPTIVE_INTEGRATOR_FUNCTION, FITZHUGHNAGUMO_INTEGRATOR_FUNCTION, FUNCTION, \
    INCREMENT, INITIALIZER, INPUT_PORTS, INTEGRATOR_FUNCTION, INTEGRATOR_FUNCTION_TYPE, \
    INTERACTIVE_ACTIVATION_INTEGRATOR_FUNCTION, LEAKY_COMPETING_INTEGRATOR_FUNCTION, \
    MULTIPLICATIVE_PARAM, NOISE, OFFSET, OPERATION, ORNSTEIN_UHLENBECK_INTEGRATOR_FUNCTION, OUTPUT_PORTS, PRODUCT, \
    RATE, REST, SIMPLE_INTEGRATOR_FUNCTION, SUM, TIME_STEP_SIZE, THRESHOLD, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.utilities import parameter_spec, all_within_range, \
    convert_all_elements_to_np_array
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
import numpy as np

import typecheck as tc

class TestIntegrator(IntegratorFunction):  # -------------------------------------------------------------------------

    componentName = 'TEST_INTEGRATOR_FUNCTION'

    class Parameters(IntegratorFunction.Parameters):

         rate = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM], function_arg=True)
         offset = Parameter(0.0, modulable=True, aliases=[ADDITIVE_PARAM], function_arg=True)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 rate: tc.optional(parameter_spec) = None,
                 noise=None,
                 offset=None,
                 initializer=None,
                 params: tc.optional(tc.optional(dict)) = None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None):


        super().__init__(
            default_variable=default_variable,
            rate=rate,
            noise=noise,
            offset=offset,
            initializer=initializer,
            params=params,
            owner=owner,
            prefs=prefs,
        )


    def _function(self,
                 variable=None,
                 context=None,
                 params=None,
                 ):

        rate = np.array(self._get_current_parameter_value(RATE, context)).astype(float)
        offset = self._get_current_parameter_value(OFFSET, context)

        # execute noise if it is a function
        noise = self._try_execute_param(self._get_current_parameter_value(NOISE, context), variable, context=context)
        previous_value = self.parameters.previous_value._get(context)
        new_value = variable

        value = previous_value + (new_value * rate) + noise

        adjusted_value = value + offset

        # If this NOT an initialization run, update the old value
        # If it IS an initialization run, leave as is
        #    (don't want to count it as an execution step)
        if not self.is_initializing:
            self.parameters.previous_value._set(adjusted_value, context)

        return self.convert_output_type(adjusted_value)

    '''
    def _gen_llvm_integrate(self, builder, index, ctx, vi, vo, params, state):
        rate = self._gen_llvm_load_param(ctx, builder, params, index, RATE)
        offset = self._gen_llvm_load_param(ctx, builder, params, index, OFFSET)
        noise = self._gen_llvm_load_param(ctx, builder, params, index, NOISE,
                                          state=state)

        # Get the only context member -- previous value
        prev_ptr = pnlvm.helpers.get_state_ptr(builder, self, state, "previous_value")
        # Get rid of 2d array. When part of a Mechanism the input,
        # (and output, and context) are 2d arrays.
        prev_ptr = pnlvm.helpers.unwrap_2d_array(builder, prev_ptr)
        assert len(prev_ptr.type.pointee) == len(vi.type.pointee)

        prev_ptr = builder.gep(prev_ptr, [ctx.int32_ty(0), index])
        prev_val = builder.load(prev_ptr)

        vi_ptr = builder.gep(vi, [ctx.int32_ty(0), index])
        vi_val = builder.load(vi_ptr)

        new_val = builder.fmul(vi_val, rate)

        ret = builder.fadd(prev_val, new_val)
        ret = builder.fadd(ret, noise)
        res = builder.fadd(ret, offset)

        vo_ptr = builder.gep(vo, [ctx.int32_ty(0), index])
        builder.store(res, vo_ptr)
        builder.store(res, prev_ptr)'''
