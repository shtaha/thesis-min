import os

import numpy as np

from MIP_oracle.agents import make_agent
from MIP_oracle.chronics import get_sorted_chronics
from MIP_oracle.dc_opf import load_case, CaseParameters
from MIP_oracle.visualizer import pprint

# Environment parameters
env_dc = True
verbose = False

# Agent parameters
kwargs = {}

for case_name in [

    #"rte_case14_realistic",
    #"rte_case5_example"
    "l2rpn_2019",
]:

    """
    Initialize environment.
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
    case = load_case(case_name, env_parameters=parameters, verbose=verbose)
    env = case.env

    for agent_name in [
        #"do-nothing-agent",
        "agent-mip",
        #"agent-multistep-mip",
    ]:
        np.random.seed(0)

        """
            Initialize agent.
        """
        # Switching penalty
        if "rte_case5" in case_name:
            kwargs["obj_lambda_action"] = 0.006
        elif "l2rpn_2019" in case_name:
            kwargs["obj_lambda_action"] = 0.07
        else:
            kwargs["obj_lambda_action"] = 0.05

        agent = make_agent(agent_name, case, **kwargs)
        agent.print_agent(default=verbose)

        chronics_dir, chronics, chronics_sorted = get_sorted_chronics(env=env)
        pprint("Chronics:", chronics_dir)

        # Ensure the ordering of chronics to be always the same
        for chronic_idx, chronic_name in enumerate(chronics_sorted):
            chronic_org_idx = chronics.index(chronic_name)
            env.chronics_handler.tell_id(chronic_org_idx - 1)  # Set chronic id

            obs = env.reset()
            agent.reset(obs=obs)

            chronic_len = env.chronics_handler.real_data.data.max_iter
            chronic_path_name = "/".join(
                os.path.normpath(env.chronics_handler.get_id()).split(os.sep)[-3:]
            )
            pprint("    - Chronic:", chronic_path_name)

            """
                Reinfocement Learning Loop
            """
            t = 0
            done = False
            reward = np.nan

            while not done:
                action = agent.act(obs, reward=reward, done=done)
                obs_next, reward, done, info = env.step(action)

                t = env.chronics_handler.real_data.data.current_index

                if t % 200 == 0 or t == 10:
                    done = True
                    pprint("        - Step:", t)

                if done:
                    pprint("        - Length:", f"{t}/{chronic_len}")

                obs = obs_next

            if chronic_idx == 1:
                break
