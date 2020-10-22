import numpy as np
import pyomo.environ as pyo

from .standard import StandardDCOPF
from ..parameters import LineSwitchingParameters


class LineSwitchingDCOPF(StandardDCOPF):
    def __init__(
        self,
        name,
        grid,
        grid_backend,
        params=LineSwitchingParameters(),
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            grid=grid,
            grid_backend=grid_backend,
            params=params,
            verbose=verbose,
            **kwargs,
        )

        # Optimal line status
        self.x = None

    """
        VARIABLES
    """

    def _build_variables(self):
        self._build_variables_standard_deltas()  # Bus voltage angles with bounds
        self._build_variables_lines()  # Power line flows without bounds
        self._build_variables_standard_generators()  # Generator productions with bounds

        if len(self.ext_grid.index):
            self._build_variables_standard_ext_grids()  # External grid productions with bounds

        self._build_variables_line_switching()

    def _build_variables_lines(self):
        # Power line flows
        self.model.line_flow = pyo.Var(
            self.model.line_set,
            domain=pyo.Reals,
            initialize=self._create_map_ids_to_values(self.line.index, self.line.p_pu),
        )

    def _build_variables_line_switching(self):
        # Power line status
        # x = 0: Line is disconnected.
        # x = 1: Line is disconnected.
        self.model.x = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.status.values.astype(int)
            ),
        )

    """
        CONSTRAINTS.
    """

    def _build_constraints(self):
        self._build_constraint_bus_balance()  # Bus power balance

        self._build_constraint_line_flows()  # Power flow definition
        self._build_constraint_line_max_flows()  # Power flow bounds with indicator variables

        self._build_constraint_max_line_status_changes()  # Power line status change

    def _build_constraint_line_flows(self):
        """
        Power line flow definition.

            big_m = False: F_l = F_ij = b_ij * (delta_i - delta_j) * x_l
            big_m = True: -M_l (1 - x_l) <= F_ij - b_ij * (delta_i - delta_j) <= M_l * (1 - x_l)

            M_l = b_l * (pi/2 - (- pi/2)) = b_l * pi
        """

        if self.params.big_m:
            self.model.big_m = pyo.Param(
                self.model.line_set,
                initialize=self._create_map_ids_to_values(
                    self.line.index, self.line.b_pu * 2 * self.params.delta_max
                ),
                within=pyo.PositiveReals,
            )

            # -M_l(1 - x_l) <= F_ij - b_ij * (delta_i - delta_j) <= M_l * (1 - x_l)
            def _constraint_line_flow_upper(model, line_id):
                return model.line_flow[line_id] - model.line_b[line_id] * (
                    model.delta[model.line_ids_to_bus_ids[line_id][0]]
                    - model.delta[model.line_ids_to_bus_ids[line_id][1]]
                ) <= model.big_m[line_id] * (1 - model.x[line_id])

            def _constraint_line_flow_lower(model, line_id):
                return -model.big_m[line_id] * (
                    1 - model.x[line_id]
                ) <= model.line_flow[line_id] - model.line_b[line_id] * (
                    model.delta[model.line_ids_to_bus_ids[line_id][0]]
                    - model.delta[model.line_ids_to_bus_ids[line_id][1]]
                )

            self.model.constraint_line_flow_upper = pyo.Constraint(
                self.model.line_set, rule=_constraint_line_flow_upper
            )

            self.model.constraint_line_flow_lower = pyo.Constraint(
                self.model.line_set, rule=_constraint_line_flow_lower
            )
        else:

            def _constraint_line_flow(model, line_id):
                return (
                    model.line_flow[line_id]
                    == model.line_b[line_id]
                    * (
                        model.delta[model.line_ids_to_bus_ids[line_id][0]]
                        - model.delta[model.line_ids_to_bus_ids[line_id][1]]
                    )
                    * model.x[line_id]
                )

            self.model.constraint_line_flow = pyo.Constraint(
                self.model.line_set, rule=_constraint_line_flow
            )

    def _build_constraint_line_max_flows(self):
        # Indicator constraints on power line flow
        # -F_l^max * x_l <= F_l <= F_l^max * x_l

        def _constraint_max_flow_lower(model, line_id):
            return (
                -model.line_flow_max[line_id] * model.x[line_id]
                <= model.line_flow[line_id]
            )

        def _constraint_max_flow_upper(model, line_id):
            return (
                model.line_flow[line_id]
                <= model.line_flow_max[line_id] * model.x[line_id]
            )

        self.model.constraint_line_max_flow_lower = pyo.Constraint(
            self.model.line_set, rule=_constraint_max_flow_lower
        )
        self.model.constraint_line_max_flow_upper = pyo.Constraint(
            self.model.line_set, rule=_constraint_max_flow_upper
        )

    def _build_constraint_max_line_status_changes(self):
        def _constraint_max_line_status_changes(model):
            line_status_change = [
                1 - model.x[line_id] if model.line_status[line_id] else model.x[line_id]
                for line_id in model.line_set
            ]

            return sum(line_status_change) <= self.params.n_max_line_status_changed

        self.model.constraint_max_line_status_changes = pyo.Constraint(
            rule=_constraint_max_line_status_changes
        )

    """
        OBJECTIVE.    
    """

    def _build_objective(self):
        # Minimize generator costs
        def _objective_gen_p(model):
            return sum(
                [
                    model.gen_p[gen_id] * self.gen.cost_pu[gen_id]
                    for gen_id in model.gen_set
                ]
            )

        # Maximize line margins
        def _objective_line_margin(model):
            return sum(
                [
                    model.line_flow[line_id] ** 2 / model.line_flow_max[line_id] ** 2
                    for line_id in model.line_set
                ]
            )

        def _objective(model):
            obj = 0

            if self.params.line_margin and self.params.solver_name != "glpk":
                obj = obj + _objective_line_margin(model)

            if (
                not (self.params.line_margin and self.params.solver_name != "glpk")
                or self.params.gen_cost
            ):
                obj = obj + _objective_gen_p(model)

            return obj

        self.model.objective = pyo.Objective(rule=_objective, sense=pyo.minimize)

    """
        SOLVE FUNCTIONS.
    """

    def solve(self, verbose=False, time_limit=5):
        self._solve(
            verbose=verbose,
            tol=self.params.tol,
            time_limit=time_limit,
            warm_start=self.params.warm_start,
        )

        # Solution status
        solution_status = self.solver_status["Solver"][0]["Termination condition"]

        # Duality gap
        lower_bound = self.solver_status["Problem"][0]["Lower bound"]
        upper_bound = self.solver_status["Problem"][0]["Upper bound"]
        gap = np.abs((upper_bound - lower_bound) / lower_bound)
        if gap < 1e-2:
            gap = 1e-2

        # Save standard DC-OPF variable results
        self._solve_save()

        # Save line status variable
        self.x = self._round_solution(self._access_pyomo_variable(self.model.x))
        self.res_line["status"] = self.x[~self.line.trafo]
        self.res_trafo["status"] = self.x[self.line.trafo]

        if verbose:
            self.model.display()

        result = {
            "res_cost": self.res_cost,
            "res_bus": self.res_bus,
            "res_line": self.res_line,
            "res_gen": self.res_gen,
            "res_x": self.x,
            "res_gap": gap,
            "solution_status": solution_status,
        }
        return result
