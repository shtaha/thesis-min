import numpy as np
import pandas as pd


class RewardL2RPN2019:
    def from_observation(self, obs):
        relative_flows = obs.rho
        reward = self.from_relative_flows(relative_flows)
        return reward

    def from_mip_solution(self, result):
        line_flow = pd.concat(
            [result["res_line"]["p_pu"], result["res_trafo"]["p_pu"]], ignore_index=True
        )
        max_line_flow = pd.concat(
            [result["res_line"]["max_p_pu"], result["res_trafo"]["max_p_pu"]],
            ignore_index=True,
        )

        relative_flows = np.abs(
            np.divide(line_flow, max_line_flow + 1e-9)
        )  # rho_l = abs(F_l / F_l^max)
        relative_flows = relative_flows * np.greater(relative_flows, 1e-9).astype(float)

        reward = self.from_relative_flows(relative_flows)
        return reward

    @staticmethod
    def from_relative_flows(relative_flows):
        relative_flows = np.minimum(relative_flows, 1.0)  # Clip if rho > 1.0

        line_scores = np.maximum(
            1.0 - relative_flows ** 2, np.zeros_like(relative_flows)
        )

        reward = line_scores.sum()
        return reward
