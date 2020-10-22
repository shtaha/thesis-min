from ..dc_opf import (
    GridDCOPF,
    TopologyOptimizationDCOPF,
    MultistepTopologyDCOPF,
    SinglestepTopologyParameters,
    MultistepTopologyParameters,
    Forecasts,
)
from ..dc_opf.rewards import RewardL2RPN2019
from ..visualizer import pprint


def make_agent(
        agent_name,
        case,
        save_dir=None,
        verbose=False,
        horizon=2,
        **kwargs,
):
    action_set = case.generate_unitary_action_set(
        case, case_save_dir=save_dir, verbose=verbose
    )

    if agent_name == "agent-multistep-mip":
        agent = AgentMultistepMIP(
            case=case, action_set=action_set, horizon=horizon, **kwargs
        )
    elif agent_name == "agent-mip":
        agent = AgentMIP(case=case, action_set=action_set, **kwargs)
    elif agent_name == "do-nothing-agent":
        agent = AgentDoNothing(case=case, action_set=action_set)
    else:
        raise ValueError(f"Agent name {agent_name} is invalid.")

    return agent


class BaseAgent:
    def __init__(self, name, case):
        self.name = name

        self.case = case
        self.env = case.env

        self.grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

    def act(self, observation, reward, done=False):
        pass

    def reset(self, obs):
        pass

    def set_kwargs(self, **kwargs):
        pass

    def get_reward(self):
        pass

    def print_agent(self, default=False):
        print("\n" + "-" * 80)
        pprint("Agent:", self.name, shift=36)
        print("-" * 80)


class AgentDoNothing(BaseAgent):
    def __init__(self, case, action_set):
        BaseAgent.__init__(self, name="Do-nothing Agent", case=case)

        self.model_kwargs = dict()

        self.reward = None
        self.obs_next = None
        self.done = None
        self.result = None

        self.actions, self.actions_info = action_set

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)
        action = self.actions[0]

        obs_next, reward, done, info = observation.simulate(action)
        self.reward = reward
        self.obs_next = obs_next
        self.done = done

        return action

    def _update(self, obs, reset=False, verbose=False):
        self.grid.update(obs, reset=reset, verbose=verbose)

    def get_reward(self):
        return self.reward


class AgentMIP(BaseAgent):
    """
    Agent class used for experimentation and testing.
    """

    def __init__(
            self,
            case,
            action_set,
            reward_class=RewardL2RPN2019,
            **kwargs,
    ):
        BaseAgent.__init__(self, name="Agent MIP", case=case)

        if "n_max_line_status_changed" not in kwargs:
            kwargs[
                "n_max_line_status_changed"
            ] = case.env.parameters.MAX_LINE_STATUS_CHANGED

        if "n_max_sub_changed" not in kwargs:
            kwargs["n_max_sub_changed"] = case.env.parameters.MAX_SUB_CHANGED

        if "n_max_timestep_overflow" not in kwargs:
            kwargs[
                "n_max_timestep_overflow"
            ] = case.env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED

        self.default_kwargs = kwargs
        self.model_kwargs = self.default_kwargs
        self.params = SinglestepTopologyParameters(**self.model_kwargs)

        self.forecasts = None
        self.reset(obs=None)

        self.model = None
        self.result = None

        self.reward_function = reward_class()
        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        self.model_kwargs = {**self.default_kwargs, **kwargs}
        self.params = SinglestepTopologyParameters(**self.model_kwargs)

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)
        self.model = TopologyOptimizationDCOPF(
            self.case.env.name,
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )

        self.model.build_model()
        self.result = self.model.solve()

        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]
        return action

    def reset(self, obs):
        if self.params.forecasts:
            self.forecasts = Forecasts(
                env=self.env,
                t=self.env.chronics_handler.real_data.data.current_index,
                horizon=1,
            )

    def _update(self, obs, reset=False, verbose=False):
        if self.params.forecasts:
            self.forecasts.t = self.forecasts.t + 1
        self.grid.update(obs, reset=reset, verbose=verbose)

    def get_reward(self):
        return self.reward_function.from_mip_solution(self.result)

    def print_agent(self, default=False):
        default_kwargs = SinglestepTopologyParameters().to_dict()

        print("\n" + "-" * 80)
        pprint("Agent:", self.name, shift=36)
        if default:
            for arg in default_kwargs:
                model_arg = self.model_kwargs[arg] if arg in self.model_kwargs else "-"
                pprint(
                    f"  - {arg}:", "{:<10}".format(str(model_arg)), default_kwargs[arg]
                )
        else:
            for arg in self.model_kwargs:
                if arg in default_kwargs:
                    pprint(
                        f"  - {arg}:",
                        "{:<10}".format(str(self.model_kwargs[arg])),
                        default_kwargs[arg],
                    )
        print("-" * 80)


class AgentMultistepMIP(BaseAgent):
    """
    Agent class used for experimentation and testing.
    """

    def __init__(
            self,
            case,
            action_set,
            reward_class=RewardL2RPN2019,
            **kwargs,
    ):
        BaseAgent.__init__(self, name="Agent Multistep MIP", case=case)

        if "n_max_line_status_changed" not in kwargs:
            kwargs[
                "n_max_line_status_changed"
            ] = case.env.parameters.MAX_LINE_STATUS_CHANGED

        if "n_max_sub_changed" not in kwargs:
            kwargs["n_max_sub_changed"] = case.env.parameters.MAX_SUB_CHANGED

        if "n_max_timestep_overflow" not in kwargs:
            kwargs[
                "n_max_timestep_overflow"
            ] = case.env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED

        self.default_kwargs = kwargs
        self.model_kwargs = self.default_kwargs
        self.params = MultistepTopologyParameters(**self.model_kwargs)

        self.forecasts = None
        self.reset(obs=None)

        self.model = None
        self.result = None

        self.reward_function = reward_class()
        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        self.model_kwargs = {**self.default_kwargs, **kwargs}
        self.params = MultistepTopologyParameters(**self.model_kwargs)

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)
        self.model = MultistepTopologyDCOPF(
            self.case.env.name,
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )

        self.model.build_model()
        self.result = self.model.solve()

        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]
        return action

    def reset(self, obs):
        if self.params.forecasts:
            self.forecasts = Forecasts(
                env=self.env,
                t=self.env.chronics_handler.real_data.data.current_index,
                horizon=self.params.horizon,
            )

    def _update(self, obs, reset=False, verbose=False):
        if self.params.forecasts:
            self.forecasts.t = self.forecasts.t + 1
        self.grid.update(obs, reset=reset, verbose=verbose)

    def get_reward(self):
        return self.reward_function.from_mip_solution(self.result)

    def print_agent(self, default=False):
        default_kwargs = MultistepTopologyParameters().to_dict()

        print("\n" + "-" * 80)
        pprint("Agent:", self.name, shift=36)
        if default:
            for arg in default_kwargs:
                model_arg = self.model_kwargs[arg] if arg in self.model_kwargs else "-"
                pprint(
                    f"  - {arg}:", "{:<10}".format(str(model_arg)), default_kwargs[arg]
                )
        else:
            for arg in self.model_kwargs:
                pprint(
                    f"  - {arg}:",
                    "{:<10}".format(str(self.model_kwargs[arg])),
                    default_kwargs[arg],
                )
        print("-" * 80)
