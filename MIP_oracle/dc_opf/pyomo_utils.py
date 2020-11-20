import numpy as np
import pyomo.environ as pyo


class PyomoMixin:
    @staticmethod
    def _round_solution(x):
        x = np.round(x)
        x = x.astype(np.bool)
        return x

    @staticmethod
    def _dataframe_to_list_of_tuples(df):
        return [tuple(row) for row in df.to_numpy()]

    @staticmethod
    def _create_map_ids_to_values_sum(ids, sum_ids, values):
        return {idx: values[list(sum_ids[idx])].sum() for idx in ids}

    @staticmethod
    def _create_map_ids_to_values(ids, values):
        return {idx: value for idx, value in zip(ids, values)}

    @staticmethod
    def _create_map_dual_ids_to_values(ids_first, ids_second, values):
        """
        Returns a dictionary, a mapping, from two sets of indices to values.

        Inputs:
            ids_first: m
            ids_second: n
            values: m x n

        Outputs:
            map[ids_first[i], ids_second[j]] = values[i, j]
        """

        mapping = dict()
        for j, idx_second in enumerate(ids_second):
            for i, idx_first in enumerate(ids_first):
                value = values[i, j]
                mapping[(idx_first, idx_second)] = value
        return mapping

    @staticmethod
    def _create_map_triple_ids_to_values(ids_first, ids_second, ids_third, values):
        """
        Returns a dictionary, a mapping, from two sets of indices to values.

        Inputs:
            ids_first: m
            ids_second: n
            values: m x n

        Outputs:
            map[ids_first[i], ids_second[j], ids_third[k]] = values[i, j, k]
        """

        mapping = dict()
        for k, idx_third in enumerate(ids_third):
            for j, idx_second in enumerate(ids_second):
                for i, idx_first in enumerate(ids_first):
                    value = values[i, j, k]
                    mapping[(idx_first, idx_second, idx_first)] = value
        return mapping

    @staticmethod
    def _access_pyomo_variable(var):
        return np.array([pyo.value(var[idx]) for idx in var])

    @staticmethod
    def _access_pyomo_dual_variable(var):
        ids_first = np.unique([idx[0] for idx in var])
        ids_second = np.unique([idx[1] for idx in var])

        values = np.empty((len(ids_first), len(ids_second)))
        for idx in var:
            value = pyo.value(var[idx])
            values[idx[0], idx[1]] = value

        return values
