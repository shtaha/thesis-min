import numpy as np
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints

from .standard import StandardDCOPF
from ..parameters import SinglestepTopologyParameters


class TopologyOptimizationDCOPF(StandardDCOPF):
    def __init__(
        self,
        name,
        grid,
        grid_backend=None,
        forecasts=None,
        params=SinglestepTopologyParameters(),
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

        self.forecasts = forecasts

        # Optimal switching status
        self.x_gen = None
        self.x_load = None
        self.x_line_or_1 = None
        self.x_line_or_2 = None
        self.x_line_ex_1 = None
        self.x_line_ex_2 = None

        # Auxiliary
        self.x_line_status_switch = None
        self.x_substation_topology_switch = None

    def build_model_do_nothing(self):
        self.build_model()
        self._build_constraint_do_nothing_action()

    """
        INDEXED SETS
    """

    def _build_indexed_sets(self):
        self._build_indexed_sets_standard()
        self._build_indexed_sets_substation()

    def _build_indexed_sets_substation(self):
        self.model.sub_set = pyo.Set(
            initialize=self.sub.index,
            within=pyo.NonNegativeIntegers,
        )
        self.model.sub_bus_set = pyo.Set(
            initialize=[1, 2],
            within=pyo.NonNegativeIntegers,
        )

    """
        PARAMETERS.
    """

    def _build_parameters_injections(self):
        init_value = (
            self.forecasts.load_p.flatten() if self.forecasts else self.load.p_pu
        )
        if init_value.shape[0] != len(self.load.index):
            init_value = self.load.p_pu

        self.model.load_p = pyo.Param(
            self.model.load_set,
            initialize=self._create_map_ids_to_values(self.load.index, init_value),
            within=pyo.Reals,
        )

        if self.params.obj_lin_gen_penalty or self.params.obj_quad_gen_penalty:
            init_value = (
                self.forecasts.prod_p.flatten() if self.forecasts else self.gen.p_pu
            )
            if init_value.shape[0] != len(self.gen.index):
                init_value = self.gen.p_pu

            self.model.gen_p_ref = pyo.Param(
                self.model.gen_set,
                initialize=self._create_map_ids_to_values(self.gen.index, init_value),
                within=pyo.Reals,
            )

    def _build_parameters_topology(self):
        self.model.sub_ids_to_bus_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.bus),
            within=self.model.bus_set * self.model.bus_set,
        )

        self.model.bus_ids_to_sub_bus_ids = pyo.Param(
            self.model.bus_set,
            initialize=self._create_map_ids_to_values(self.bus.index, self.bus.sub_bus),
        )

        self.model.bus_ids_to_sub_ids = pyo.Param(
            self.model.bus_set,
            initialize=self._create_map_ids_to_values(self.bus.index, self.bus["sub"]),
            within=self.model.sub_set,
        )

        if len(self.ext_grid.index):
            self.model.bus_ids_to_ext_grid_ids = pyo.Param(
                self.model.bus_set,
                initialize=self._create_map_ids_to_values(
                    self.bus.index, self.bus.ext_grid
                ),
                within=pyo.Any,
            )

        self.model.line_ids_to_sub_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index.values,
                self._dataframe_to_list_of_tuples(self.line[["sub_or", "sub_ex"]]),
            ),
            within=self.model.sub_set * self.model.sub_set,
        )

        # Line statuses
        self.model.line_status = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.status
            ),
            within=pyo.Boolean,
        )

        # Substation grid elements
        self.model.sub_ids_to_gen_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.gen),
            within=pyo.Any,
        )
        self.model.sub_ids_to_load_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.load),
            within=pyo.Any,
        )
        self.model.sub_ids_to_line_or_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.line_or),
            within=pyo.Any,
        )
        self.model.sub_ids_to_line_ex_ids = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(self.sub.index, self.sub.line_ex),
            within=pyo.Any,
        )
        self.model.sub_n_elements = pyo.Param(
            self.model.sub_set,
            initialize=self._create_map_ids_to_values(
                self.sub.index, self.sub.n_elements
            ),
            within=pyo.NonNegativeIntegers,
        )

        # Bus within a substation
        self.model.gen_ids_to_sub_bus_ids = pyo.Param(
            self.model.gen_set,
            initialize=self._create_map_ids_to_values(self.gen.index, self.gen.sub_bus),
            within=self.model.sub_bus_set,
        )
        self.model.load_ids_to_sub_bus_ids = pyo.Param(
            self.model.load_set,
            initialize=self._create_map_ids_to_values(
                self.load.index, self.load.sub_bus
            ),
            within=self.model.sub_bus_set,
        )
        self.model.line_or_ids_to_sub_bus_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.sub_bus_or
            ),
            within=self.model.sub_bus_set,
        )
        self.model.line_ex_ids_to_sub_bus_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.sub_bus_ex
            ),
            within=self.model.sub_bus_set,
        )

        if len(self.ext_grid.index):
            self.model.ext_grid_ids_to_sub_bus_ids = pyo.Param(
                self.model.ext_grid_set,
                initialize=self._create_map_ids_to_values(
                    self.ext_grid.index, self.ext_grid.sub_bus
                ),
                within=self.model.sub_bus_set,
            )

    """
        VARIABLES.
    """

    def _build_variables(self):
        self._build_variables_standard_deltas()  # Bus voltage angles with bounds

        self._build_variables_generators()  # Generator productions with bounds
        if len(self.ext_grid.index):
            self._build_variables_standard_ext_grids()  # External grid productions with bounds

        self._build_variables_lines()  # Power line flows without bounds

        # Topology optimization variables
        self._build_variables_bus_configuration()

        # Auxiliary
        self._build_variables_changes()

        self._build_variable_mu()
        self._build_variable_overflow()

        if self.params.obj_lin_gen_penalty:
            self._build_variable_mu_gen()

    def _build_variables_generators(self):
        self.model.gen_p = pyo.Var(
            self.model.gen_set,
            domain=pyo.NonNegativeReals,
            initialize=self._create_map_ids_to_values(
                self.gen.index, np.maximum(self.gen.p_pu, 0.0)
            ),
        )

        self.model.gen_sub_bus_p = pyo.Var(
            self.model.gen_set,
            self.model.sub_bus_set,
            domain=pyo.NonNegativeReals,
        )

    def _build_variables_lines(self):
        self.model.line_flow = pyo.Var(
            self.model.line_set,
            domain=pyo.Reals,
            initialize=self._create_map_ids_to_values(self.line.index, self.line.p_pu),
        )

        self.model.line_sub_bus_flow = pyo.Var(
            self.model.line_set,
            self.model.sub_bus_set,
            self.model.sub_bus_set,
            domain=pyo.Reals,
        )

    def _build_variables_bus_configuration(self):
        # Indicator variables for bus configuration of power lines, generators, and loads

        # Power line OR bus switching
        self.model.x_line_or_1 = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.equal(self.bus.sub_bus.values[self.line.bus_or.values], 1).astype(
                    int
                ),
            ),
        )
        self.model.x_line_or_2 = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.equal(self.bus.sub_bus.values[self.line.bus_or.values], 2).astype(
                    int
                ),
            ),
        )

        # Power line EX bus switching
        self.model.x_line_ex_1 = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.equal(self.bus.sub_bus.values[self.line.bus_ex.values], 1).astype(
                    int
                ),
            ),
        )
        self.model.x_line_ex_2 = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.equal(self.bus.sub_bus.values[self.line.bus_ex.values], 2).astype(
                    int
                ),
            ),
        )

        # Generator switching
        self.model.x_gen = pyo.Var(
            self.model.gen_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.gen.index,
                np.equal(self.bus.sub_bus.values[self.gen.bus.values], 2).astype(int),
            ),
        )

        # Load switching
        self.model.x_load = pyo.Var(
            self.model.load_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.load.index,
                np.equal(self.bus.sub_bus.values[self.load.bus.values], 2).astype(int),
            ),
        )

    def _build_variables_changes(self):
        # Auxiliary variables for counting the number of line status changes and substation topology reconfigurations
        self.model.x_line_status_switch = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.zeros_like(self.line.index),
            ),
        )

        self.model.x_substation_topology_switch = pyo.Var(
            self.model.sub_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.sub.index,
                np.zeros_like(self.sub.index),
            ),
        )

    def _build_variable_mu(self):
        init_value = np.abs(self.line.p_pu) / self.line.max_p_pu
        init_value[np.isnan(init_value)] = 0.0

        value = init_value.copy()
        value[np.greater_equal(init_value, 1.0)] = 0.0
        self.model.mu = pyo.Var(
            self.model.line_set,
            domain=pyo.NonNegativeReals,
            bounds=(0.0, 1.0),
            initialize=self._create_map_ids_to_values(self.line.index, value),
        )

        value = init_value.copy()
        value[np.less_equal(init_value, 1.0)] = 0.0
        self.model.mu_overflow = pyo.Var(
            self.model.line_set,
            domain=pyo.NonNegativeReals,
            bounds=(0.0, self.grid.max_rho),
            initialize=self._create_map_ids_to_values(self.line.index, value),
        )

        self.model.mu_max = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0.0, self.grid.max_rho),
            initialize=np.max(init_value),
        )

    def _build_variable_overflow(self):
        self.model.f_line = pyo.Var(
            self.model.line_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                np.greater(self.line.p_pu, self.line.max_p_pu).astype(int),
            ),
        )

    def _build_variable_mu_gen(self):
        self.model.mu_gen = pyo.Var(
            domain=pyo.NonNegativeReals, bounds=(0.0, 1.0), initialize=1.0
        )

    """
        CONSTRAINTS.
    """

    def _build_constraints(self):
        self._build_constraint_generators()  # Generator production with bounds

        self._build_constraint_line_flows()  # Power flow definition with loose bounds

        self._build_constraint_bus_balance()  # Bus power balance
        self._build_constraint_line_or()
        self._build_constraint_line_ex()

        if not self.params.con_allow_onesided_disconnection:
            self._build_constraint_onesided_line_disconnection()

        if not self.params.con_allow_onesided_reconnection:
            self._build_constraint_onesided_line_reconnection()

        if self.params.con_symmetry:
            self._build_constraint_symmetry()

        if self.params.con_requirement_at_least_two:
            self._build_constraint_requirement_at_least_two()

        if self.params.con_requirement_balance:
            self._build_constraint_requirement_balance()

        if self.params.con_switching_limits:
            self._build_constraint_line_status_switch()
            self._build_constraint_substation_topology_switch()

        if self.params.con_cooldown:
            self._build_constraint_cooldown()

        if self.params.con_overflow:
            self._build_constraint_overflow()

        if self.params.con_maintenance:
            self._build_constraint_maintenance()

        if self.params.con_unitary_action:
            self._build_constraint_unitary_action()

        if self.params.obj_lin_gen_penalty:
            self._build_constraint_lin_gen_penalty()

    def _build_constraint_generators(self):
        self.__build_constraint_generators_sum()
        self.__build_constraint_generators_bounds()

    def __build_constraint_generators_sum(self):
        def _constraint_genenerators_p_sum(model, gen_id):
            total_prod = 0
            for sub_bus in model.sub_bus_set:
                prod = model.gen_sub_bus_p[gen_id, sub_bus]
                total_prod = total_prod + prod

            return model.gen_p[gen_id] == total_prod

        self.model.constraint_genenerators_p_sum = pyo.Constraint(
            self.model.gen_set, rule=_constraint_genenerators_p_sum
        )

    def __build_constraint_generators_bounds(self):
        def _constraint_genenerators_p_bounds_lower(model, gen_id, sub_bus):
            if sub_bus == 1:
                x = 1 - model.x_gen[gen_id]
            else:
                x = model.x_gen[gen_id]

            return model.gen_p_min[gen_id] * x <= model.gen_sub_bus_p[gen_id, sub_bus]

        def _constraint_genenerators_p_bounds_upper(model, gen_id, sub_bus):
            if sub_bus == 1:
                x = 1 - model.x_gen[gen_id]
            else:
                x = model.x_gen[gen_id]

            return model.gen_sub_bus_p[gen_id, sub_bus] <= model.gen_p_max[gen_id] * x

        self.model.constraint_genenerators_p_bounds_lower = pyo.Constraint(
            self.model.gen_set,
            self.model.sub_bus_set,
            rule=_constraint_genenerators_p_bounds_lower,
        )

        self.model.constraint_genenerators_p_bounds_upper = pyo.Constraint(
            self.model.gen_set,
            self.model.sub_bus_set,
            rule=_constraint_genenerators_p_bounds_upper,
        )

    def _build_constraint_line_flows(self):
        self.__build_constraint_line_flows_sum()  # Actual power flow in a line
        self.__build_constraint_line_sub_bus_flows()  # Power flow definition
        self.__build_constraint_line_sub_bus_flow_bounds()  # Sub bus power flow thermal limits
        self.__build_constraint_line_flow_bounds()  # Line power flow thermal limits

        self.__build_constraint_line_overflow()  # Power flow bounds, underflow and overflow
        self.__build_constraint_mu_max()  # Maximum relative flow

    def __build_constraint_line_flows_sum(self):
        # Power flow equation with topology switching
        def _constraint_line_flow_sum(model, line_id):
            total_flow = 0.0
            for sub_bus_or in model.sub_bus_set:
                for sub_bus_ex in model.sub_bus_set:
                    flow = model.line_sub_bus_flow[line_id, sub_bus_or, sub_bus_ex]
                    total_flow = total_flow + flow
            return model.line_flow[line_id] == total_flow

        self.model.constraint_line_flow_sum = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_flow_sum
        )

    def __build_constraint_line_sub_bus_flows(self):
        # Power flow equation with topology switching
        def _constraint_line_flow_lower(model, line_id, sub_bus_or, sub_bus_ex):
            sub_or, sub_ex = model.line_ids_to_sub_ids[line_id]
            bus_or = model.sub_ids_to_bus_ids[sub_or][sub_bus_or - 1]
            bus_ex = model.sub_ids_to_bus_ids[sub_ex][sub_bus_ex - 1]

            if sub_bus_or == 1:
                x_or = model.x_line_or_1[line_id]
            else:
                x_or = model.x_line_or_2[line_id]

            if sub_bus_ex == 1:
                x_ex = model.x_line_ex_1[line_id]
            else:
                x_ex = model.x_line_ex_2[line_id]

            big_m = model.line_b[line_id] * 2 * self.params.delta_max
            return model.line_sub_bus_flow[line_id, sub_bus_or, sub_bus_ex] - (
                model.line_b[line_id] * (model.delta[bus_or] - model.delta[bus_ex])
            ) <= big_m * (2 - x_or - x_ex)

        def _constraint_line_flow_upper(model, line_id, sub_bus_or, sub_bus_ex):
            sub_or, sub_ex = model.line_ids_to_sub_ids[line_id]
            bus_or = model.sub_ids_to_bus_ids[sub_or][sub_bus_or - 1]
            bus_ex = model.sub_ids_to_bus_ids[sub_ex][sub_bus_ex - 1]

            if sub_bus_or == 1:
                x_or = model.x_line_or_1[line_id]
            else:
                x_or = model.x_line_or_2[line_id]

            if sub_bus_ex == 1:
                x_ex = model.x_line_ex_1[line_id]
            else:
                x_ex = model.x_line_ex_2[line_id]

            big_m = model.line_b[line_id] * 2 * self.params.delta_max
            return -big_m * (2 - x_or - x_ex) <= model.line_sub_bus_flow[
                line_id, sub_bus_or, sub_bus_ex
            ] - (model.line_b[line_id] * (model.delta[bus_or] - model.delta[bus_ex]))

        self.model.constraint_line_flow_lower = pyo.Constraint(
            self.model.line_set,
            self.model.sub_bus_set,
            self.model.sub_bus_set,
            rule=_constraint_line_flow_lower,
        )

        self.model.constraint_line_flow_upper = pyo.Constraint(
            self.model.line_set,
            self.model.sub_bus_set,
            self.model.sub_bus_set,
            rule=_constraint_line_flow_upper,
        )

    def __build_constraint_line_sub_bus_flow_bounds(self):
        def _constraint_line_flow_bounds_or_lower(
            model, line_id, sub_bus_or, sub_bus_ex
        ):
            if sub_bus_or == 1:
                x_or = model.x_line_or_1[line_id]
            else:
                x_or = model.x_line_or_2[line_id]

            return (
                -self.grid.max_rho * model.line_flow_max[line_id] * x_or
                <= model.line_sub_bus_flow[line_id, sub_bus_or, sub_bus_ex]
            )

        def _constraint_line_flow_bounds_or_upper(
            model, line_id, sub_bus_or, sub_bus_ex
        ):
            if sub_bus_or == 1:
                x_or = model.x_line_or_1[line_id]
            else:
                x_or = model.x_line_or_2[line_id]

            return (
                model.line_sub_bus_flow[line_id, sub_bus_or, sub_bus_ex]
                <= self.grid.max_rho * model.line_flow_max[line_id] * x_or
            )

        def _constraint_line_flow_bounds_ex_lower(
            model, line_id, sub_bus_or, sub_bus_ex
        ):
            if sub_bus_ex == 1:
                x_ex = model.x_line_ex_1[line_id]
            else:
                x_ex = model.x_line_ex_2[line_id]

            return (
                -self.grid.max_rho * model.line_flow_max[line_id] * x_ex
                <= model.line_sub_bus_flow[line_id, sub_bus_or, sub_bus_ex]
            )

        def _constraint_line_flow_bounds_ex_upper(
            model, line_id, sub_bus_or, sub_bus_ex
        ):
            if sub_bus_ex == 1:
                x_ex = model.x_line_ex_1[line_id]
            else:
                x_ex = model.x_line_ex_2[line_id]

            return (
                model.line_sub_bus_flow[line_id, sub_bus_or, sub_bus_ex]
                <= self.grid.max_rho * model.line_flow_max[line_id] * x_ex
            )

        self.model.constraint_line_flow_bounds_or_lower = pyo.Constraint(
            self.model.line_set,
            self.model.sub_bus_set,
            self.model.sub_bus_set,
            rule=_constraint_line_flow_bounds_or_lower,
        )
        self.model.constraint_line_flow_bounds_or_upper = pyo.Constraint(
            self.model.line_set,
            self.model.sub_bus_set,
            self.model.sub_bus_set,
            rule=_constraint_line_flow_bounds_or_upper,
        )

        self.model.constraint_line_flow_bounds_ex_lower = pyo.Constraint(
            self.model.line_set,
            self.model.sub_bus_set,
            self.model.sub_bus_set,
            rule=_constraint_line_flow_bounds_ex_lower,
        )
        self.model.constraint_line_flow_bounds_ex_upper = pyo.Constraint(
            self.model.line_set,
            self.model.sub_bus_set,
            self.model.sub_bus_set,
            rule=_constraint_line_flow_bounds_ex_upper,
        )

    def __build_constraint_line_flow_bounds(self):
        def _constraint_line_flow_bounds_upper(model, line_id):
            return model.line_flow[line_id] <= model.line_flow_max[line_id] * (
                model.mu[line_id] + model.mu_overflow[line_id]
            )

        def _constraint_line_flow_bounds_lower(model, line_id):
            return (
                -model.line_flow_max[line_id]
                * (model.mu[line_id] + model.mu_overflow[line_id])
                <= model.line_flow[line_id]
            )

        self.model.constraint_line_flow_bounds_upper = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_flow_bounds_upper
        )

        self.model.constraint_line_flow_bounds_lower = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_flow_bounds_lower
        )

    def __build_constraint_line_overflow(self):
        def _constraint_mu_not_overflow(model, line_id):
            return model.mu[line_id] <= 1.0 - model.f_line[line_id]

        def _constraint_mu_overflow(model, line_id):
            return (
                model.mu_overflow[line_id] <= self.grid.max_rho * model.f_line[line_id]
            )

        self.model.constraint_mu_not_overflow = pyo.Constraint(
            self.model.line_set, rule=_constraint_mu_not_overflow
        )

        self.model.constraint_mu_overflow = pyo.Constraint(
            self.model.line_set, rule=_constraint_mu_overflow
        )

    def __build_constraint_mu_max(self):
        def _constraint_mu_mu_max(model, line_id):
            return model.mu[line_id] <= model.mu_max

        def _constraint_mu_overflow_mu_max(model, line_id):
            return model.mu_overflow[line_id] <= model.mu_max

        self.model.constraint_mu_mu_max = pyo.Constraint(
            self.model.line_set, rule=_constraint_mu_mu_max
        )

        self.model.constraint_mu_overflow_mu_max = pyo.Constraint(
            self.model.line_set, rule=_constraint_mu_overflow_mu_max
        )

    def _build_constraint_bus_balance(self):
        # Bus power balance constraints
        def _constraint_bus_balance(model, bus_id):
            sub_id = model.bus_ids_to_sub_ids[bus_id]
            sub_bus = model.bus_ids_to_sub_bus_ids[bus_id]

            # Generator bus injections
            bus_gen_ids = model.sub_ids_to_gen_ids[sub_id]
            if len(bus_gen_ids):
                bus_gen_p = [
                    model.gen_sub_bus_p[gen_id, sub_bus] for gen_id in bus_gen_ids
                ]
                sum_gen_p = sum(bus_gen_p)
            else:
                sum_gen_p = 0.0

            if len(self.ext_grid.index):
                bus_ext_grid_ids = model.bus_ids_to_ext_grid_ids[bus_id]
                bus_ext_grids_p = [
                    model.ext_grid_p[ext_grid_id] for ext_grid_id in bus_ext_grid_ids
                ]
                sum_gen_p = sum_gen_p + sum(bus_ext_grids_p)

            # Load bus injections
            bus_load_ids = model.sub_ids_to_load_ids[sub_id]
            if len(bus_load_ids):
                bus_load_p = [
                    model.load_p[load_id] * (1 - model.x_load[load_id])
                    if sub_bus == 1
                    else model.load_p[load_id] * model.x_load[load_id]
                    for load_id in bus_load_ids
                ]
                sum_load_p = sum(bus_load_p)
            else:
                sum_load_p = 0.0

            # Power line flows
            flows_out = [
                sum(
                    [
                        model.line_sub_bus_flow[line_id, sub_bus, sub_bus_ex]
                        for sub_bus_ex in model.sub_bus_set
                    ]
                )
                for line_id in model.line_set
                if sub_id == model.line_ids_to_sub_ids[line_id][0]
            ]

            flows_in = [
                sum(
                    [
                        model.line_sub_bus_flow[line_id, sub_bus_or, sub_bus]
                        for sub_bus_or in model.sub_bus_set
                    ]
                )
                for line_id in model.line_set
                if sub_id == model.line_ids_to_sub_ids[line_id][1]
            ]

            if len(flows_in) == 0 and len(flows_out) == 0:
                return pyo.Constraint.Skip

            return sum_gen_p - sum_load_p == sum(flows_out) - sum(flows_in)

        self.model.constraint_bus_balance = pyo.Constraint(
            self.model.bus_set, rule=_constraint_bus_balance
        )

    def _build_constraint_line_or(self):
        def _constraint_line_or(model, line_id):
            return model.x_line_or_1[line_id] + model.x_line_or_2[line_id] <= 1

        self.model.constraint_line_or = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_or
        )

    def _build_constraint_line_ex(self):
        def _constraint_line_ex(model, line_id):
            return model.x_line_ex_1[line_id] + model.x_line_ex_2[line_id] <= 1

        self.model.constraint_line_ex = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_ex
        )

    def _build_constraint_onesided_line_disconnection(self):
        def _constraint_onesided_line_disconnection(model, line_id):
            return (
                model.x_line_or_1[line_id] + model.x_line_or_2[line_id]
                == model.x_line_ex_1[line_id] + model.x_line_ex_2[line_id]
            )

        self.model.constraint_onesided_line_disconnection = pyo.Constraint(
            self.model.line_set, rule=_constraint_onesided_line_disconnection
        )

    def _build_constraint_onesided_line_reconnection(self):
        for sub_id in self.model.sub_set:
            lines_or = self.model.sub_ids_to_line_or_ids[sub_id]
            lines_ex = self.model.sub_ids_to_line_ex_ids[sub_id]

            lines_or_disconnected = [
                not self.model.line_status[line_id] for line_id in lines_or
            ]
            lines_ex_disconnected = [
                not self.model.line_status[line_id] for line_id in lines_ex
            ]

            if any(lines_or_disconnected) or any(lines_ex_disconnected):
                self.model.x_substation_topology_switch[sub_id].fix(0)
                self.model.x_substation_topology_switch[sub_id].setlb(0)
                self.model.x_substation_topology_switch[sub_id].setub(0)

    def _build_constraint_symmetry(self):
        for sub_id in self.grid.fixed_elements.index:
            line_or = self.grid.fixed_elements.line_or[sub_id]
            line_ex = self.grid.fixed_elements.line_ex[sub_id]

            if len(line_or):
                line_id = line_or[0]
                self.model.x_line_or_2[line_id].fix(0)
                self.model.x_line_or_2[line_id].setlb(0)
                self.model.x_line_or_2[line_id].setub(0)

            if len(line_ex):
                line_id = line_ex[0]
                self.model.x_line_ex_2[line_id].fix(0)
                self.model.x_line_ex_2[line_id].setlb(0)
                self.model.x_line_ex_2[line_id].setub(0)

    def _build_constraint_requirement_at_least_two(self):
        self.model.w_bus_activation = pyo.Var(
            self.model.bus_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.bus.index, np.greater(self.bus.n_elements, 1.0).astype(int)
            ),
        )

        def _get_bus_elements(model, bus_id):
            sub_id = model.bus_ids_to_sub_ids[bus_id]
            sub_bus = model.bus_ids_to_sub_bus_ids[bus_id]

            gens = [
                1 - model.x_gen[gen_id] if sub_bus == 1 else model.x_gen[gen_id]
                for gen_id in model.sub_ids_to_gen_ids[sub_id]
            ]
            loads = [
                1 - model.x_load[load_id] if sub_bus == 1 else model.x_load[load_id]
                for load_id in model.sub_ids_to_load_ids[sub_id]
            ]

            lines_or = [
                model.x_line_or_1[line_id]
                if sub_bus == 1
                else model.x_line_or_2[line_id]
                for line_id in model.sub_ids_to_line_or_ids[sub_id]
            ]
            lines_ex = [
                model.x_line_ex_1[line_id]
                if sub_bus == 1
                else model.x_line_ex_2[line_id]
                for line_id in model.sub_ids_to_line_ex_ids[sub_id]
            ]
            return sub_id, sub_bus, gens, loads, lines_or, lines_ex

        def _constraint_implicit_line_disconnection_lower(model, bus_id):
            sub_id, sub_bus, gens, loads, lines_or, lines_ex = _get_bus_elements(
                model, bus_id
            )

            return 2 * model.w_bus_activation[bus_id] <= sum(gens) + sum(loads) + sum(
                lines_or
            ) + sum(lines_ex)

        def _constraint_implicit_line_disconnection_upper(model, bus_id):
            sub_id, sub_bus, gens, loads, lines_or, lines_ex = _get_bus_elements(
                model, bus_id
            )
            n_elements = model.sub_n_elements[sub_id]

            return (
                sum(gens) + sum(loads) + sum(lines_or) + sum(lines_ex)
                <= n_elements * model.w_bus_activation[bus_id]
            )

        self.model.constraint_implicit_line_disconnection_lower = pyo.Constraint(
            self.model.bus_set, rule=_constraint_implicit_line_disconnection_lower
        )

        self.model.constraint_implicit_line_disconnection_upper = pyo.Constraint(
            self.model.bus_set, rule=_constraint_implicit_line_disconnection_upper
        )

    def _build_constraint_requirement_balance(self):
        self.model.w_bus_balance = pyo.Var(
            self.model.bus_set,
            domain=pyo.Binary,
            initialize=self._create_map_ids_to_values(
                self.bus.index,
                [
                    np.greater(
                        len(self.bus.gen[bus_id]) + len(self.bus.load[bus_id]), 0
                    ).astype(int)
                    for bus_id in self.bus.index
                ],
            ),
        )

        def _get_bus_elements(model, bus_id):
            sub_id = model.bus_ids_to_sub_ids[bus_id]
            sub_bus = model.bus_ids_to_sub_bus_ids[bus_id]

            gens = [
                1 - model.x_gen[gen_id] if sub_bus == 1 else model.x_gen[gen_id]
                for gen_id in model.sub_ids_to_gen_ids[sub_id]
            ]
            loads = [
                1 - model.x_load[load_id] if sub_bus == 1 else model.x_load[load_id]
                for load_id in model.sub_ids_to_load_ids[sub_id]
            ]

            lines_or = [
                model.x_line_or_1[line_id]
                if sub_bus == 1
                else model.x_line_or_2[line_id]
                for line_id in model.sub_ids_to_line_or_ids[sub_id]
            ]
            lines_ex = [
                model.x_line_ex_1[line_id]
                if sub_bus == 1
                else model.x_line_ex_2[line_id]
                for line_id in model.sub_ids_to_line_ex_ids[sub_id]
            ]
            return sub_id, sub_bus, gens, loads, lines_or, lines_ex

        def _constraint_gen_load_bus_lower(model, bus_id):
            _, _, gens, loads, _, _ = _get_bus_elements(model, bus_id)
            if len(gens) or len(loads):
                return model.w_bus_balance[bus_id] <= sum(gens) + sum(loads)
            else:
                return pyo.Constraint.Skip

        def _constraint_gen_load_bus_upper(model, bus_id):
            _, _, gens, loads, _, _ = _get_bus_elements(model, bus_id)
            if len(gens) or len(loads):
                return (
                    sum(gens) + sum(loads)
                    <= (len(gens) + len(loads)) * model.w_bus_balance[bus_id]
                )
            else:
                return pyo.Constraint.Skip

        def _constraint_at_least_one_line(model, bus_id):
            _, _, _, _, lines_or, lines_ex = _get_bus_elements(model, bus_id)
            return model.w_bus_balance[bus_id] <= sum(lines_or) + sum(lines_ex)

        self.model.constraint_at_least_one_line = pyo.Constraint(
            self.model.bus_set, rule=_constraint_at_least_one_line
        )

        self.model.constraint_gen_load_bus_lower = pyo.Constraint(
            self.model.bus_set, rule=_constraint_gen_load_bus_lower
        )

        self.model.constraint_gen_load_bus_upper = pyo.Constraint(
            self.model.bus_set, rule=_constraint_gen_load_bus_upper
        )

    def _build_constraint_cooldown(self):
        for line_id in self.line.index:
            if self.line.cooldown[line_id] > 0:
                self.model.x_line_status_switch[line_id].fix(0)
                self.model.x_line_status_switch[line_id].setlb(0)
                self.model.x_line_status_switch[line_id].setub(0)

        for sub_id in self.sub.index:
            if self.sub.cooldown[sub_id] > 0:
                self.model.x_substation_topology_switch[sub_id].fix(0)
                self.model.x_substation_topology_switch[sub_id].setlb(0)
                self.model.x_substation_topology_switch[sub_id].setub(0)

    def _build_constraint_overflow(self):
        # Disconnect a power line if too long in overflow
        self.model.overflow = pyo.ConstraintList()

        for line_id in self.line.index:
            if self.params.n_max_timestep_overflow - self.line.t_overflow[line_id] == 1:
                expr = (
                    self.model.x_line_or_1[line_id]
                    + self.model.x_line_or_2[line_id]
                    + self.model.x_line_ex_1[line_id]
                    + self.model.x_line_ex_2[line_id]
                )
                self.model.overflow.add(expr == 0)

    def _build_constraint_maintenance(self):
        for line_id in self.line.index:
            t_main = self.line.next_maintenance[line_id]
            d_main = self.line.duration_maintenance[line_id]
            in_main = t_main == 0 and d_main > 0

            if in_main:  # Under maintenance
                self.model.x_line_status_switch[line_id].fix(0)
                self.model.x_line_status_switch[line_id].setlb(0)
                self.model.x_line_status_switch[line_id].setub(0)
            else:  # Not under maintenance
                if (
                    t_main < 0 or d_main == 0
                ):  # If no planned maintenance or duration is zero
                    pass
                else:
                    if t_main == 1:  # Only if mainentance at the next time step
                        self.model.x_line_status_switch[line_id].fix(0)
                        self.model.x_line_status_switch[line_id].setlb(0)
                        self.model.x_line_status_switch[line_id].setub(0)

    def _build_constraint_unitary_action(self):
        def _constraint_unitary_action(model):
            x_line = sum(
                [model.x_line_status_switch[line_id] for line_id in model.line_set]
            )
            x_sub = sum(
                [model.x_substation_topology_switch[sub_id] for sub_id in model.sub_set]
            )
            return x_line + x_sub <= 1

        self.model.constraint_unitary_action = pyo.Constraint(
            rule=_constraint_unitary_action
        )

    def _build_constraint_line_status_switch(self):
        def _constraint_line_status_switch(model, line_id):
            if model.line_status[line_id]:
                x_line_status = 1 - model.x_line_status_switch[line_id]
            else:
                x_line_status = model.x_line_status_switch[line_id]

            return (
                model.x_line_or_1[line_id]
                + model.x_line_or_2[line_id]
                + model.x_line_ex_1[line_id]
                + model.x_line_ex_2[line_id]
                == 2 * x_line_status
            )

        def _constraint_max_line_status_switch(model):
            return (
                sum([model.x_line_status_switch[line_id] for line_id in model.line_set])
                <= self.params.n_max_line_status_changed
            )

        # Auxiliary constraint for checking line status switch
        self.model.constraint_line_status_switch = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_status_switch
        )

        # Limit the number of line status switches
        self.model.constraint_max_line_status_switch = pyo.Constraint(
            rule=_constraint_max_line_status_switch
        )

    def _build_constraint_substation_topology_switch(self):
        def _get_substation_switch_terms(model, sub_id):
            sub_gen_ids = model.sub_ids_to_gen_ids[sub_id]
            sub_load_ids = model.sub_ids_to_load_ids[sub_id]
            sub_line_or_ids = model.sub_ids_to_line_or_ids[sub_id]
            sub_line_ex_ids = model.sub_ids_to_line_ex_ids[sub_id]

            x_sub_gen_switch = [
                model.x_gen[gen_id]
                if model.gen_ids_to_sub_bus_ids[gen_id] == 1
                else 1 - model.x_gen[gen_id]
                for gen_id in sub_gen_ids
            ]
            x_sub_load_switch = [
                model.x_load[load_id]
                if model.load_ids_to_sub_bus_ids[load_id] == 1
                else 1 - model.x_load[load_id]
                for load_id in sub_load_ids
            ]

            x_sub_line_or_switch = []
            for line_id in sub_line_or_ids:
                if model.line_status[line_id]:
                    if model.line_or_ids_to_sub_bus_ids[line_id] == 1:
                        x_sub_line_or_switch.append(model.x_line_or_2[line_id])
                    elif model.line_or_ids_to_sub_bus_ids[line_id] == 2:
                        x_sub_line_or_switch.append(model.x_line_or_1[line_id])
                    else:
                        raise ValueError("No such substation bus.")
                else:
                    # If line is reconnected, then skip
                    pass

            x_sub_line_ex_switch = []
            for line_id in sub_line_ex_ids:
                if model.line_status[line_id]:
                    if model.line_ex_ids_to_sub_bus_ids[line_id] == 1:
                        x_sub_line_ex_switch.append(model.x_line_ex_2[line_id])
                    elif model.line_ex_ids_to_sub_bus_ids[line_id] == 2:
                        x_sub_line_ex_switch.append(model.x_line_ex_1[line_id])
                    else:
                        raise ValueError("No such substation bus.")
                else:
                    # If line is reconnected, then skip
                    pass
            return (
                x_sub_gen_switch,
                x_sub_load_switch,
                x_sub_line_or_switch,
                x_sub_line_ex_switch,
            )

        def _constraint_substation_topology_switch_upper(model, sub_id):
            (
                x_sub_gen_switch,
                x_sub_load_switch,
                x_sub_line_or_switch,
                x_sub_line_ex_switch,
            ) = _get_substation_switch_terms(model, sub_id)
            return (
                sum(x_sub_gen_switch)
                + sum(x_sub_load_switch)
                + sum(x_sub_line_or_switch)
                + sum(x_sub_line_ex_switch)
                <= model.sub_n_elements[sub_id]
                * model.x_substation_topology_switch[sub_id]
            )

        def _constraint_substation_topology_switch_lower(model, sub_id):
            (
                x_sub_gen_switch,
                x_sub_load_switch,
                x_sub_line_or_switch,
                x_sub_line_ex_switch,
            ) = _get_substation_switch_terms(model, sub_id)

            return (
                sum(x_sub_gen_switch)
                + sum(x_sub_load_switch)
                + sum(x_sub_line_or_switch)
                + sum(x_sub_line_ex_switch)
                >= model.x_substation_topology_switch[sub_id]
            )

        def _constraint_max_substation_topology_switch(model):
            return (
                sum(
                    [
                        model.x_substation_topology_switch[sub_id]
                        for sub_id in model.sub_set
                    ]
                )
                <= self.params.n_max_sub_changed
            )

        # Auxiliary constraint for checking substation topology reconfigurations
        self.model.constraint_substation_topology_switch_lower = pyo.Constraint(
            self.model.sub_set, rule=_constraint_substation_topology_switch_lower
        )
        self.model.constraint_substation_topology_switch_upper = pyo.Constraint(
            self.model.sub_set, rule=_constraint_substation_topology_switch_upper
        )

        # Limit the number of substation topology reconfigurations
        self.model.constraint_max_substation_topology_switch = pyo.Constraint(
            rule=_constraint_max_substation_topology_switch
        )

    def _build_constraint_lin_gen_penalty(self):
        def _constraint_lin_gen_penalty_upper(model, gen_id):
            return (model.gen_p[gen_id] - model.gen_p_ref[gen_id]) / model.gen_p_max[
                gen_id
            ] <= model.mu_gen

        def _constraint_lin_gen_penalty_lower(model, gen_id):
            return (
                -model.mu_gen
                <= (model.gen_p[gen_id] - model.gen_p_ref[gen_id])
                / model.gen_p_max[gen_id]
            )

        self.model.constraint_lin_gen_penalty_lower = pyo.Constraint(
            self.model.gen_set, rule=_constraint_lin_gen_penalty_upper
        )

        self.model.constraint_lin_gen_penalty_upper = pyo.Constraint(
            self.model.gen_set, rule=_constraint_lin_gen_penalty_lower
        )

    def _build_constraint_do_nothing_action(self):
        def _constraint_do_nothing_action(model):
            line_switch = sum(
                [model.x_line_status_switch[line_id] for line_id in model.line_set]
            )
            sub_switch = sum(
                [model.x_substation_topology_switch[sub_id] for sub_id in model.sub_set]
            )
            return line_switch + sub_switch <= 0.0

        self.model.constraint_do_nothing_action = pyo.Constraint(
            rule=_constraint_do_nothing_action
        )

    """
        OBJECTIVE.
    """

    def _build_objective(self):
        assert (
            self.params.obj_gen_cost
            or self.params.obj_reward_lin
            or self.params.obj_reward_quad
            or self.params.obj_reward_max
            or self.params.obj_lin_gen_penalty
            or self.params.obj_quad_gen_penalty
        )

        assert (
            not (self.params.obj_reward_lin and self.params.obj_reward_quad)
            and not (self.params.obj_reward_lin and self.params.obj_reward_max)
            and not (self.params.obj_reward_max and self.params.obj_reward_quad)
        )  # Only one penalty on margins
        assert not (
            self.params.obj_lin_gen_penalty and self.params.obj_quad_gen_penalty
        )  # Only one penalty on generators

        """
            Generator power production cost. As in standard OPF.
        """

        def _objective_gen_p(model):
            return sum(
                [
                    model.gen_p[gen_id] * self.gen.cost_pu[gen_id]
                    for gen_id in model.gen_set
                ]
            )

        """
            Line margins.
        """

        # Linear
        def _objective_lin_line_margins(model):
            penalty = 0.0
            for line_id in model.line_set:
                penalty = penalty + (model.mu[line_id] + model.f_line[line_id])
            return penalty / len(model.line_set)

        # Quadratic
        def _objective_quad_line_margins(model):
            return sum(
                [
                    model.line_flow[line_id] ** 2 / model.line_flow_max[line_id] ** 2
                    for line_id in model.line_set
                ]
            ) / len(model.line_set)

        def _objective_max_line_margins(model):
            return model.mu_max

        """
            Generator power production error.
        """

        # Linear penalty on generator power productions
        def _objective_lin_gen_penalty(model):
            return self.params.obj_lambda_gen * model.mu_gen

        # Quadratic penalty on generator power productions
        def _objective_quad_gen_penalty(model):
            penalty = sum(
                [
                    (
                        (model.gen_p[gen_id] - model.gen_p_ref[gen_id])
                        / (model.gen_p_max[gen_id])
                    )
                    ** 2
                    for gen_id in model.gen_set
                ]
            )

            return self.params.obj_lambda_gen / len(model.gen_set) * penalty

        """
            Penalize actions. Prefer do-nothing actions.
        """

        def _objective_action_penalty(model):
            penalty = 0
            if model.x_line_status_switch:
                penalty = penalty + sum(
                    [model.x_line_status_switch[line_id] for line_id in model.line_set]
                )
            if model.x_substation_topology_switch:
                penalty = penalty + sum(
                    [
                        model.x_substation_topology_switch[sub_id]
                        for sub_id in model.sub_set
                    ]
                )

            return self.params.obj_lambda_action * penalty

        def _objective(model):
            obj = 0.0
            if self.params.obj_gen_cost:
                obj = obj + _objective_gen_p(model)

            if self.params.obj_reward_lin:
                obj = obj + _objective_lin_line_margins(model)
            elif self.params.obj_reward_quad:
                obj = obj + _objective_quad_line_margins(model)
            elif self.params.obj_reward_max:
                obj = obj + _objective_max_line_margins(model)

            if self.params.obj_lin_gen_penalty:
                obj = obj + _objective_lin_gen_penalty(model)
            elif self.params.obj_quad_gen_penalty:
                obj = obj + _objective_quad_gen_penalty(model)

            if self.params.obj_lambda_action > 0.0:
                obj = obj + _objective_action_penalty(model)

            return obj

        self.model.objective = pyo.Objective(rule=_objective, sense=pyo.minimize)

    """
        SOLVE FUNCTIONS.
    """

    def solve(self, verbose=False):
        self._solve(
            verbose=verbose,
            tol=self.params.tol,
            warm_start=self.params.warm_start,
            time_limit=self.params.time_limit,
        )

        # Solution status
        solution_status = self.solver_status["Solver"][0]["Termination condition"]
        if verbose and solution_status == "infeasible":
            log_infeasible_constraints(
                self.model, tol=self.params.tol, log_expression=False
            )

        # Duality gap
        lower_bound, upper_bound, gap = 0.0, 0.0, 0.0
        if solution_status != "infeasible":
            lower_bound = self.solver_status["Problem"][0]["Lower bound"]
            upper_bound = self.solver_status["Problem"][0]["Upper bound"]
            gap = np.minimum(
                np.abs((upper_bound - lower_bound) / (lower_bound + 1e-9)), 0.1
            )

        if gap < 1e-4:
            gap = 1e-4

        # Save standard DC-OPF variable results
        self._solve_save()

        # Save line status variable
        self.x_gen = self._round_solution(self._access_pyomo_variable(self.model.x_gen))
        self.x_load = self._round_solution(
            self._access_pyomo_variable(self.model.x_load)
        )
        self.x_line_or_1 = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_or_1)
        )
        self.x_line_or_2 = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_or_2)
        )
        self.x_line_ex_1 = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_ex_1)
        )
        self.x_line_ex_2 = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_ex_2)
        )

        self.x_line_status_switch = self._round_solution(
            self._access_pyomo_variable(self.model.x_line_status_switch)
        )
        self.x_substation_topology_switch = self._round_solution(
            self._access_pyomo_variable(self.model.x_substation_topology_switch)
        )

        if verbose:
            self.model.display()

        result = {
            "res_cost": self.res_cost,
            "res_bus": self.res_bus,
            "res_line": self.res_line,
            "res_gen": self.res_gen,
            "res_load": self.res_load,
            "res_ext_grid": self.res_ext_grid,
            "res_trafo": self.res_trafo,
            "res_x": np.concatenate(
                (
                    self.x_gen,
                    self.x_load,
                    self.x_line_or_1,
                    self.x_line_or_2,
                    self.x_line_ex_1,
                    self.x_line_ex_2,
                )
            ),
            "res_x_gen": self.x_gen,
            "res_x_load": self.x_load,
            "res_x_line_or_1": self.x_line_or_1,
            "res_x_line_or_2": self.x_line_or_2,
            "res_x_line_ex_1": self.x_line_ex_1,
            "res_x_line_ex_2": self.x_line_ex_2,
            "res_x_line_status_switch": self.x_line_status_switch,
            "res_x_substation_topology_switch": self.x_substation_topology_switch,
            "res_gap": gap,
            "solution_status": solution_status,
        }

        return result
