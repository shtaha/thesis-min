import os
from abc import ABC, abstractmethod

import grid2op
import numpy as np
import pandas as pd
from grid2op.Environment import Environment

from .unit_converter import UnitConverter
from ..action_space import ActionSpaceGenerator
from ..constants import Constants as Const
from ..visualizer import describe_environment


def load_case(case_name, env_parameters=None, verbose=False):
    if "rte_case5" in case_name:
        case = OPFRTECase5(case_name=case_name, env_parameters=env_parameters)
    elif "l2rpn_2019" in case_name:
        case = OPFL2RPN2019(case_name=case_name, env_parameters=env_parameters)
    elif "l2rpn_wcci_2020" in case_name:
        case = OPFL2RPN2020(case_name=case_name, env_parameters=env_parameters)
    else:
        raise ValueError(f"Invalid case name. Case {case_name} does not exist.")

    if verbose and case.env:
        describe_environment(case.env)
    elif case.env:
        env_pf = "DC" if case.env.parameters.ENV_DC else "AC"
        print(f"\n{case.env.name.upper()} ({env_pf})\n")

    return case


class OPFAbstractCase(ABC):
    @abstractmethod
    def build_case_grid(self):
        pass


class OPFCaseMixin:
    @staticmethod
    def _update_backend(env, grid):
        n_line = len(grid.line.index)

        # Grid element names from environment names
        grid.line["name"] = env.name_line[0:n_line]
        grid.gen["name"] = env.name_gen
        grid.load["name"] = env.name_load
        grid.trafo["name"] = env.name_line[n_line:]

        # Update thermal limits with environment thermal limits
        grid.line["max_i_ka"] = env.get_thermal_limit()[0:n_line] / 1000.0

        # Environment and backend inconsistency
        grid.gen["min_p_mw"] = env.gen_pmin
        grid.gen["max_p_mw"] = env.gen_pmax
        grid.gen["type"] = env.gen_type
        grid.gen["gen_redispatchable"] = env.gen_redispatchable
        grid.gen["gen_max_ramp_up"] = env.gen_max_ramp_up
        grid.gen["gen_max_ramp_down"] = env.gen_max_ramp_down
        grid.gen["gen_min_uptime"] = env.gen_min_uptime
        grid.gen["gen_min_downtime"] = env.gen_min_downtime

    @staticmethod
    def make_environment(case_name, parameters):
        if parameters:
            env: Environment = grid2op.make_from_dataset_path(
                dataset_path=os.path.join(Const.DATASET_DIR, case_name),
                backend=grid2op.Backend.PandaPowerBackend(),
                action_class=grid2op.Action.TopologyAction,
                observation_class=grid2op.Observation.CompleteObservation,
                reward_class=grid2op.Reward.L2RPNReward,
                param=parameters,
            )
        else:
            env = grid2op.make(case_name)
        return env

    @staticmethod
    def generate_unitary_action_set(case, case_save_dir=None, verbose=False):
        action_generator = ActionSpaceGenerator(case.env)
        action_set = action_generator.get_topology_action_set(
            save_dir=case_save_dir, verbose=verbose
        )
        return action_set


class OPFRTECase5(OPFAbstractCase, UnitConverter, OPFCaseMixin):
    def __init__(self, case_name="rte_case5_example", env_parameters=None):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=1e5)

        self.name = "Case RTE 5"

        self.env = self.make_environment(case_name=case_name, parameters=env_parameters)

        self.grid_org = self.build_case_grid()
        self.grid_backend = self.update_backend(self.env)
        self.env.backend._grid = self.grid_backend

    def build_case_grid(self):
        return self.env.backend._grid.deepcopy()

    def update_backend(self, env):
        """
        Update backend grid with missing data.
        """
        grid = env.backend._grid.deepcopy()

        # Check if even number of buses
        assert len(grid.bus.index) % 2 == 0

        # Bus names
        bus_names = []
        for bus_id, bus_name in zip(grid.bus.index, grid.bus["name"]):
            sub_id = bus_name.split("_")[-1]
            bus_names.append(f"bus-{bus_id}-{sub_id}")
        grid.bus["name"] = bus_names

        # Controllable injections
        grid.load["controllable"] = False
        grid.gen["controllable"] = True

        self._update_backend(env, grid)

        return grid


class OPFL2RPN2019(OPFAbstractCase, UnitConverter, OPFCaseMixin):
    def __init__(self, case_name="l2rpn_2019", env_parameters=None):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=1e5)

        self.name = "Case L2RPN 2019"

        self.env = self.make_environment(case_name=case_name, parameters=env_parameters)

        self.grid_org = self.build_case_grid()
        self.grid_backend = self.update_backend(self.env)
        self.env.backend._grid = self.grid_backend

    def build_case_grid(self):
        return self.env.backend._grid.deepcopy()

    def update_backend(self, env):
        """
        Update backend grid with missing data.
        """
        grid = env.backend._grid.deepcopy()

        # Check if even number of buses
        assert len(grid.bus.index) % 2 == 0

        # Bus names
        bus_names = [
            f"bus-{bus_id}-{sub_id}"
            for bus_id, sub_id in zip(grid.bus.index, grid.bus["name"])
        ]
        grid.bus["name"] = bus_names

        self._update_backend(env, grid)

        return grid


class OPFL2RPN2020(OPFAbstractCase, UnitConverter, OPFCaseMixin):
    def __init__(self, case_name="l2rpn_wcci_2020", env_parameters=None):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=138000.0)

        self.name = "Case L2RPN 2020 WCCI"

        self.env = self.make_environment(case_name=case_name, parameters=env_parameters)

        self.grid_org = self.build_case_grid()
        self.grid_backend = self.update_backend(self.env)
        self.env.backend._grid = self.grid_backend

    def build_case_grid(self):
        return self.env.backend._grid.deepcopy()

    def update_backend(self, env):
        """
        Update backend grid with missing data.
        """
        grid = env.backend._grid.deepcopy()

        # Bus names
        n_sub = len(grid.bus.index) // 2
        bus_to_sub_ids = np.concatenate((np.arange(0, n_sub), np.arange(0, n_sub)))
        bus_names = [
            f"bus-{bus_id}-{sub_id}"
            for bus_id, sub_id in zip(grid.bus.index, bus_to_sub_ids)
        ]
        grid.bus["name"] = bus_names

        self._update_backend(env, grid)

        # Manually set
        trafo_params = {
            "id": {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
            },
            "b_pu": {
                "0": 2852.04991087,
                "1": 2698.61830743,
                "2": 3788.16577013,
                "3": 2890.59112589,
            },
            "max_p_pu": {"0": 9900.0, "1": 9900.0, "2": 9900.0, "3": 9900.0},
        }

        trafo_params = pd.DataFrame.from_dict(trafo_params)
        trafo_params.set_index("id", inplace=True)
        grid.trafo["b_pu"] = trafo_params["b_pu"]
        grid.trafo["max_p_pu"] = trafo_params["max_p_pu"]

        return grid
