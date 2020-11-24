import sys
from abc import ABC

from grid2op.Parameters import Parameters


class CaseParameters(Parameters):
    def __init__(self, case_name, env_dc=False):
        Parameters.__init__(self, parameters_path=None)

        param_dict = self._get_param_dict(case_name=case_name)

        self.init_from_dict(dict_=param_dict)
        if env_dc:
            self.ENV_DC = env_dc
            self.FORECAST_DC = env_dc

    @staticmethod
    def _get_param_dict(case_name):
        """
        Copied from grid2op config files.
        """

        if "rte_case5" in case_name:
            param_dict = {
                "NO_OVERFLOW_DISCONNECTION": False,
                "IGNORE_MIN_UP_DOWN_TIME": True,
                "ALLOW_DISPATCH_GEN_SWITCH_OFF": True,
                "NB_TIMESTEP_OVERFLOW_ALLOWED": 2,
                "NB_TIMESTEP_RECONNECTION": 10,
                "HARD_OVERFLOW_THRESHOLD": 2.0,
                "ENV_DC": False,
                "FORECAST_DC": False,
                "MAX_SUB_CHANGED": 1,
                "MAX_LINE_STATUS_CHANGED": 1,
                "NB_TIMESTEP_COOLDOWN_LINE": 0,
                "NB_TIMESTEP_COOLDOWN_SUB": 0,
            }
        elif "l2rpn_2019" in case_name:
            param_dict = {
                "NO_OVERFLOW_DISCONNECTION": False,
                "IGNORE_MIN_UP_DOWN_TIME": True,
                "ALLOW_DISPATCH_GEN_SWITCH_OFF": True,
                "NB_TIMESTEP_OVERFLOW_ALLOWED": 2,
                "NB_TIMESTEP_RECONNECTION": 10,
                "HARD_OVERFLOW_THRESHOLD": 2.0,
                "ENV_DC": False,
                "FORECAST_DC": False,
                "MAX_SUB_CHANGED": 1,
                "MAX_LINE_STATUS_CHANGED": 1,
                "NB_TIMESTEP_COOLDOWN_LINE": 0,
                "NB_TIMESTEP_COOLDOWN_SUB": 0,
            }
        else:
            # 1) changed this for all other cases
            param_dict = {
                "NO_OVERFLOW_DISCONNECTION": False,
                "IGNORE_MIN_UP_DOWN_TIME": True,
                "ALLOW_DISPATCH_GEN_SWITCH_OFF": True,
                "NB_TIMESTEP_OVERFLOW_ALLOWED": 3,
                "NB_TIMESTEP_RECONNECTION": 12,
                "HARD_OVERFLOW_THRESHOLD": 200.0,
                "ENV_DC": False,
                "FORECAST_DC": False,
                "MAX_SUB_CHANGED": 1,
                "MAX_LINE_STATUS_CHANGED": 1,
                "NB_TIMESTEP_COOLDOWN_LINE": 3,
                "NB_TIMESTEP_COOLDOWN_SUB": 3,
            }
        return param_dict


class AbstractParameters(ABC):
    def to_dict(self):
        return self.__dict__


class SolverParameters(AbstractParameters):
    def __init__(
        self,
        solver_name="gurobi",
        tol=0.0001,
        warm_start=False,
        time_limit=5,
    ):
        #if sys.platform != "win32":
        #    solver_name = "glpk"

        self.solver_name = solver_name
        self.tol = tol
        self.warm_start = warm_start
        self.time_limit = time_limit


class StandardParameters(SolverParameters):
    def __init__(self, delta_max=0.5, **kwargs):
        SolverParameters.__init__(self, **kwargs)
        self.delta_max = delta_max


class LineSwitchingParameters(StandardParameters):
    def __init__(
        self,
        n_max_line_status_changed=1,
        big_m=True,
        gen_cost=True,
        line_margin=True,
        time_limit=7,
        **kwargs,
    ):
        StandardParameters.__init__(self, time_limit=time_limit, **kwargs)

        self.n_max_line_status_changed = n_max_line_status_changed

        self.big_m = big_m

        self.gen_cost = gen_cost
        self.line_margin = line_margin


class SinglestepTopologyParameters(StandardParameters):
    def __init__(
        self,
        forecasts=True,
        n_max_line_status_changed=1,
        n_max_sub_changed=1,
        n_max_timestep_overflow=2,
        con_allow_onesided_disconnection=False,
        con_allow_onesided_reconnection=False,
        con_symmetry=True,
        con_requirement_at_least_two=True,
        con_requirement_balance=True,
        con_switching_limits=True,
        con_cooldown=True,
        con_overflow=True,
        con_maintenance=True,
        con_unitary_action=False,
        obj_gen_cost=False,
        obj_reward_lin=False,
        obj_reward_quad=False,
        obj_reward_max=True,
        obj_lambda_gen=100.0,
        obj_lin_gen_penalty=True,
        obj_quad_gen_penalty=False,
        obj_lambda_action=0.0,
        time_limit=7,
        **kwargs,
    ):
        StandardParameters.__init__(self, time_limit=time_limit, **kwargs)

        self.forecasts = forecasts

        self.n_max_line_status_changed = n_max_line_status_changed
        self.n_max_sub_changed = n_max_sub_changed
        self.n_max_timestep_overflow = n_max_timestep_overflow

        self.con_allow_onesided_disconnection = con_allow_onesided_disconnection
        self.con_allow_onesided_reconnection = con_allow_onesided_reconnection
        self.con_symmetry = con_symmetry
        self.con_requirement_at_least_two = con_requirement_at_least_two
        self.con_requirement_balance = con_requirement_balance

        self.con_switching_limits = con_switching_limits
        self.con_cooldown = con_cooldown
        self.con_overflow = con_overflow
        self.con_maintenance = con_maintenance
        self.con_unitary_action = con_unitary_action

        self.obj_gen_cost = obj_gen_cost

        self.obj_reward_lin = obj_reward_lin
        self.obj_reward_quad = obj_reward_quad
        self.obj_reward_max = obj_reward_max

        self.obj_lambda_gen = obj_lambda_gen
        self.obj_lin_gen_penalty = obj_lin_gen_penalty
        self.obj_quad_gen_penalty = obj_quad_gen_penalty

        self.obj_lambda_action = obj_lambda_action


class MultistepTopologyParameters(SinglestepTopologyParameters):
    def __init__(
        self,
        horizon=2,
        con_allow_onesided_disconnection=False,
        time_limit=20,
        **kwargs,
    ):
        SinglestepTopologyParameters.__init__(
            self,
            con_allow_onesided_disconnection=con_allow_onesided_disconnection,
            time_limit=time_limit,
            **kwargs,
        )
        self.horizon = horizon
