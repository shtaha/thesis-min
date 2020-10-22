import numpy as np
import pandapower as pp
import pandas as pd
import pyomo.environ as pyo
import pyomo.opt as pyo_opt

from ..parameters import StandardParameters
from ..pyomo_utils import PyomoMixin
from ..unit_converter import UnitConverter


class StandardDCOPF(UnitConverter, PyomoMixin):
    def __init__(
        self,
        name,
        grid,
        grid_backend=None,
        params=StandardParameters(),
        verbose=False,
        **kwargs,
    ):
        UnitConverter.__init__(self, **kwargs)
        if verbose:
            self.print_base_units()

        self.name = name
        self.grid = grid
        self.grid_backend = grid_backend

        self.sub = grid.sub
        self.bus = grid.bus
        self.line = grid.line
        self.gen = grid.gen
        self.load = grid.load
        self.ext_grid = grid.ext_grid
        self.trafo = grid.trafo

        self.model = None

        self.params = params
        self.solver = pyo_opt.SolverFactory(self.params.solver_name)
        self.solver_status = None

        # Results
        self._initialize_results()

    def _initialize_results(self):
        self.res_cost = 0.0
        self.res_bus = pd.DataFrame(
            columns=["v_pu", "delta_pu", "delta_deg"], index=self.bus.index
        )
        self.res_line = pd.DataFrame(
            columns=[
                "bus_or",
                "bus_ex",
                "p_pu",
                "max_p_pu",
                "loading_percent",
                "status",
            ],
            index=self.line.index,
        )
        self.res_gen = pd.DataFrame(
            columns=["min_p_pu", "p_pu", "max_p_pu", "cost_pu"], index=self.gen.index
        )
        self.res_load = pd.DataFrame(columns=["p_pu"], index=self.load.index)
        self.res_ext_grid = pd.DataFrame(columns=["p_pu"], index=self.ext_grid.index)
        self.res_trafo = pd.DataFrame(
            columns=["p_pu", "max_p_pu", "loading_percent", "status"],
            index=self.trafo.index,
        )

    def build_model(self):
        # Model
        self.model = pyo.ConcreteModel(self.name)

        # Indexed sets
        self._build_indexed_sets()

        # Parameters
        self._build_parameters()

        # Variables
        self._build_variables()

        # Constraints
        self._build_constraints()

        # Objective
        self._build_objective()

    """
        INDEXED SETS.
    """

    def _build_indexed_sets(self):
        self._build_indexed_sets_standard()

    def _build_indexed_sets_standard(self):
        """
        Indexing over buses, lines, generators, and loads.
        """

        self.model.bus_set = pyo.Set(
            initialize=self.bus.index, within=pyo.NonNegativeIntegers
        )
        self.model.line_set = pyo.Set(
            initialize=self.line.index, within=pyo.NonNegativeIntegers
        )
        self.model.gen_set = pyo.Set(
            initialize=self.gen.index, within=pyo.NonNegativeIntegers
        )
        self.model.load_set = pyo.Set(
            initialize=self.load.index, within=pyo.NonNegativeIntegers
        )

        if len(self.ext_grid.index):
            self.model.ext_grid_set = pyo.Set(
                initialize=self.ext_grid.index, within=pyo.NonNegativeIntegers
            )

    """
        PARAMETERS.
    """

    def _build_parameters(self):
        # Fixed
        self._build_parameters_deltas()  # Bus voltage angle bounds and reference node
        self._build_parameters_generators()  # Bounds on generator production
        if len(self.ext_grid.index):
            self._build_parameters_ext_grids()  # External grid power limits
        self._build_parameters_lines()  # Power line thermal limit and susceptance

        # Variable
        self._build_parameters_topology()  # Topology of generators and power lines
        self._build_parameters_injections()  # Bus load injections

    def _build_parameters_deltas(self):
        # Bus voltage angles
        self.model.delta_max = pyo.Param(
            initialize=self.params.delta_max, within=pyo.NonNegativeReals
        )

        # Slack bus index
        self.model.slack_bus_id = pyo.Param(
            initialize=self.grid.slack_bus,
            within=self.model.bus_set,
        )

    def _build_parameters_generators(self):
        """
        Initialize generator parameters: lower and upper limits on generator power production.
        """
        self.model.gen_p_max = pyo.Param(
            self.model.gen_set,
            initialize=self._create_map_ids_to_values(
                self.gen.index, self.gen.max_p_pu
            ),
            within=pyo.NonNegativeReals,
        )

        self.model.gen_p_min = pyo.Param(
            self.model.gen_set,
            initialize=self._create_map_ids_to_values(
                self.gen.index, self.gen.min_p_pu
            ),
            within=pyo.NonNegativeReals,
        )

    def _build_parameters_lines(self):
        """
        Initialize power line parameters: power line flow thermal limits, susceptances, and line statuses.
        """
        self.model.line_flow_max = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.max_p_pu
            ),
            within=pyo.NonNegativeReals,
        )
        self.model.line_b = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(self.line.index, self.line.b_pu),
            within=pyo.NonNegativeReals,
        )

    def _build_parameters_ext_grids(self):
        self.model.ext_grid_p_max = pyo.Param(
            self.model.ext_grid_set,
            initialize=self._create_map_ids_to_values(
                self.ext_grid.index, self.ext_grid.max_p_pu
            ),
            within=pyo.NonNegativeReals,
        )
        self.model.ext_grid_p_min = pyo.Param(
            self.model.ext_grid_set,
            initialize=self._create_map_ids_to_values(
                self.ext_grid.index, self.ext_grid.min_p_pu
            ),
            within=pyo.Reals,
        )

    def _build_parameters_injections(self):
        # Load bus injections
        self.model.bus_load_p = pyo.Param(
            self.model.bus_set,
            initialize=self._create_map_ids_to_values_sum(
                self.bus.index,
                self.bus.load,
                self.load.p_pu,
            ),
            within=pyo.Reals,
        )

    def _build_parameters_topology(self):
        self.model.line_ids_to_bus_ids = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index,
                self._dataframe_to_list_of_tuples(self.line[["bus_or", "bus_ex"]]),
            ),
            within=self.model.bus_set * self.model.bus_set,
        )

        self.model.bus_ids_to_gen_ids = pyo.Param(
            self.model.bus_set,
            initialize=self._create_map_ids_to_values(self.bus.index, self.bus.gen),
            within=pyo.Any,
        )

        if len(self.ext_grid.index):
            self.model.bus_ids_to_ext_grid_ids = pyo.Param(
                self.model.bus_set,
                initialize=self._create_map_ids_to_values(
                    self.bus.index, self.bus.ext_grid
                ),
                within=pyo.Any,
            )

        self.model.line_status = pyo.Param(
            self.model.line_set,
            initialize=self._create_map_ids_to_values(
                self.line.index, self.line.status
            ),
            within=pyo.Boolean,
        )

    """
        VARIABLES.
    """

    def _build_variables(self):
        self._build_variables_standard_deltas()  # Bus voltage angles with bounds
        self._build_variables_standard_lines()  # Power line flows without bounds
        self._build_variables_standard_generators()  # Generator productions with bounds
        if len(self.ext_grid.index):
            self._build_variables_standard_ext_grids()  # External grid productions with bounds

    def _build_variables_standard_deltas(self):
        # Bus voltage angle
        def _bounds_delta(model, bus_id):
            if bus_id == pyo.value(model.slack_bus_id):
                return 0.0, 0.0
            else:
                return -model.delta_max, model.delta_max

        self.model.delta = pyo.Var(
            self.model.bus_set,
            domain=pyo.Reals,
            bounds=_bounds_delta,
            initialize=self._create_map_ids_to_values(
                self.bus.index, np.zeros_like(self.bus.index.values)
            ),
        )

    def _build_variables_standard_lines(self):
        # Line power flows
        def _bounds_flow_max(model, line_id):
            if model.line_status[line_id]:
                return -model.line_flow_max[line_id], model.line_flow_max[line_id]
            else:
                return 0.0, 0.0

        self.model.line_flow = pyo.Var(
            self.model.line_set,
            domain=pyo.Reals,
            bounds=_bounds_flow_max,
            initialize=self._create_map_ids_to_values(self.line.index, self.line.p_pu),
        )

    def _build_variables_standard_generators(self):
        def _bounds_gen_p(model, gen_id):
            return model.gen_p_min[gen_id], model.gen_p_max[gen_id]

        self.model.gen_p = pyo.Var(
            self.model.gen_set,
            domain=pyo.NonNegativeReals,
            bounds=_bounds_gen_p,
            initialize=self._create_map_ids_to_values(
                self.gen.index, np.maximum(self.gen.p_pu, 0.0)
            ),
        )

    def _build_variables_standard_ext_grids(self):
        def _bounds_ext_grid_p(model, ext_grid_id):
            return model.ext_grid_p_min[ext_grid_id], model.ext_grid_p_max[ext_grid_id]

        self.model.ext_grid_p = pyo.Var(
            self.model.ext_grid_set,
            domain=pyo.Reals,
            bounds=_bounds_ext_grid_p,
            initialize=self._create_map_ids_to_values(
                self.ext_grid.index, self.ext_grid.p_pu
            ),
        )

    """
        CONSTRAINTS.
    """

    def _build_constraints(self):
        self._build_constraint_line_flows()  # Power flow definition
        self._build_constraint_bus_balance()  # Bus power balance

    def _build_constraint_bus_balance(self):
        # Bus power balance constraints
        def _constraint_bus_balance(model, bus_id):
            bus_gen_ids = model.bus_ids_to_gen_ids[bus_id]
            bus_gen_p = [model.gen_p[gen_id] for gen_id in bus_gen_ids]

            # Injections
            sum_gen_p = 0
            if len(bus_gen_p):
                sum_gen_p = sum(bus_gen_p)

            # Add external grids to generator injections
            if len(self.ext_grid.index):
                bus_ext_grid_ids = model.bus_ids_to_ext_grid_ids[bus_id]
                bus_ext_grids_p = [
                    model.ext_grid_p[ext_grid_id] for ext_grid_id in bus_ext_grid_ids
                ]
                sum_gen_p = sum_gen_p + sum(bus_ext_grids_p)

            sum_load_p = float(model.bus_load_p[bus_id])

            # Power line flows
            flows_out = [
                model.line_flow[line_id]
                for line_id in model.line_set
                if bus_id == model.line_ids_to_bus_ids[line_id][0]
            ]

            flows_in = [
                model.line_flow[line_id]
                for line_id in model.line_set
                if bus_id == model.line_ids_to_bus_ids[line_id][1]
            ]

            if len(flows_in) == 0 and len(flows_out) == 0:
                return pyo.Constraint.Skip

            return sum_gen_p - sum_load_p == sum(flows_out) - sum(flows_in)

        self.model.constraint_bus_balance = pyo.Constraint(
            self.model.bus_set, rule=_constraint_bus_balance
        )

    def _build_constraint_line_flows(self):
        # Power flow equation
        def _constraint_line_flow(model, line_id):
            return model.line_flow[line_id] == model.line_b[line_id] * (
                model.delta[model.line_ids_to_bus_ids[line_id][0]]
                - model.delta[model.line_ids_to_bus_ids[line_id][1]]
            )

        self.model.constraint_line_flow = pyo.Constraint(
            self.model.line_set, rule=_constraint_line_flow
        )

    """
        OBJECTIVE.
    """

    def _build_objective(self):
        self._build_objective_standard()

    def _build_objective_standard(self):
        # Minimize generator costs
        def _objective(model):
            return sum(
                [
                    model.gen_p[gen_id] * self.gen.cost_pu[gen_id]
                    for gen_id in model.gen_set
                ]
            )

        self.model.objective = pyo.Objective(rule=_objective, sense=pyo.minimize)

    """
        SOLVE FUNCTIONS.
    """

    def _solve_save(self):
        # Objective
        self.res_cost = pyo.value(self.model.objective)

        # Buses
        self.res_bus = self.bus[["v_pu"]].copy()
        self.res_bus["delta_pu"] = self._access_pyomo_variable(self.model.delta)
        self.res_bus["delta_deg"] = self.convert_rad_to_deg(
            self._access_pyomo_variable(self.model.delta)
        )

        # Generators
        self.res_gen = self.gen[["min_p_pu", "max_p_pu", "cost_pu"]].copy()
        self.res_gen["p_pu"] = self._access_pyomo_variable(self.model.gen_p)

        # Power lines
        self.res_line = self.line[~self.line.trafo][
            ["bus_or", "bus_ex", "max_p_pu"]
        ].copy()
        self.res_line["p_pu"] = self._access_pyomo_variable(self.model.line_flow)[
            ~self.line.trafo
        ]
        self.res_line["loading_percent"] = np.abs(
            self.res_line["p_pu"]
            / (self.line[~self.line.trafo]["max_p_pu"] + 1e-9)
            * 100.0
        )

        # Loads
        self.res_load = self.load[["p_pu"]]

        # External grids
        if len(self.ext_grid.index):
            self.res_ext_grid["p_pu"] = self._access_pyomo_variable(
                self.model.ext_grid_p
            )

        # Transformers
        if len(self.trafo.index):
            self.res_trafo["p_pu"] = self._access_pyomo_variable(self.model.line_flow)[
                self.line.trafo
            ]

            self.res_trafo["loading_percent"] = np.abs(
                self.res_trafo["p_pu"] / (self.trafo["max_p_pu"] + 1e-9) * 100.0
            )
            self.res_trafo["max_p_pu"] = self.grid.trafo["max_p_pu"]

    def _solve_save_backend(self):
        # Convert NaNs of inactive buses to 0
        self.grid_backend.res_bus = self.grid_backend.res_bus.fillna(0)
        self.grid_backend.res_line = self.grid_backend.res_line.fillna(0)
        self.grid_backend.res_gen = self.grid_backend.res_gen.fillna(0)
        self.grid_backend.res_load = self.grid_backend.res_load.fillna(0)
        self.grid_backend.res_ext_grid = self.grid_backend.res_ext_grid.fillna(0)
        self.grid_backend.res_trafo = self.grid_backend.res_trafo.fillna(0)

        # Buses
        self.grid_backend.res_bus["delta_pu"] = self.convert_degree_to_rad(
            self.grid_backend.res_bus["va_degree"]
        )

        # Power lines
        self.grid_backend.res_line["p_pu"] = self.convert_mw_to_per_unit(
            self.grid_backend.res_line["p_from_mw"]
        )
        self.grid_backend.res_line["max_p_pu"] = self.grid.line["max_p_pu"]

        # Generators
        self.grid_backend.res_gen["p_pu"] = self.convert_mw_to_per_unit(
            self.grid_backend.res_gen["p_mw"]
        )
        self.grid_backend.res_gen["min_p_pu"] = self.convert_mw_to_per_unit(
            self.grid_backend.gen["min_p_mw"]
        )
        self.grid_backend.res_gen["max_p_pu"] = self.convert_mw_to_per_unit(
            self.grid_backend.gen["max_p_mw"]
        )
        self.grid_backend.res_gen["cost_pu"] = self.gen["cost_pu"]

        # Loads
        self.grid_backend.res_load["p_pu"] = self.convert_mw_to_per_unit(
            self.grid_backend.res_load["p_mw"]
        )

        # External grids
        self.grid_backend.res_ext_grid["p_pu"] = self.convert_mw_to_per_unit(
            self.grid_backend.res_ext_grid["p_mw"]
        )

        # Transformers
        self.grid_backend.res_trafo["p_pu"] = self.convert_mw_to_per_unit(
            self.grid_backend.res_trafo["p_hv_mw"]
        )
        self.grid_backend.res_trafo["max_p_pu"] = self.grid.trafo["max_p_pu"]

    def _solve(self, verbose=False, tol=1e-9, time_limit=5, warm_start=False):
        """
        Set options to solver and solve the MIP.
        Compatible with Gurobi, GLPK, and Mosek.

        Gurobi parameters: https://www.gurobi.com/documentation/9.0/refman/parameters.html
        """
        if self.params.solver_name == "gurobi":
            options = {
                "OptimalityTol": tol,
                "MIPGap": tol,
                "TimeLimit": time_limit,
            }
        elif self.params.solver_name == "mosek":
            options = {
                "dparam.basis_rel_tol_s": tol,
                "dparam.mio_rel_gap_const": tol,
                "dparam.optimizer_max_time": time_limit,
            }
        elif self.params.solver_name == "glpk":
            options = {
                "tmlim": time_limit,
                "mipgap": tol,
            }
        else:
            options = dict()

        if self.params.solver_name != "glpk":
            self.solver_status = self.solver.solve(
                self.model, tee=verbose, options=options, warmstart=warm_start
            )
        else:
            self.solver_status = self.solver.solve(
                self.model, tee=verbose, options=options
            )

    def solve(self, verbose=False):
        self._solve(
            verbose=verbose,
            tol=self.params.tol,
            warm_start=self.params.warm_start,
            time_limit=self.params.time_limit,
        )

        # Save standard DC-OPF variable results
        self._solve_save()

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
        }
        return result

    def solve_backend(self, verbose=False):
        for gen_id in self.gen.index.values:
            pp.create_poly_cost(
                self.grid_backend,
                gen_id,
                "gen",
                cp1_eur_per_mw=self.convert_per_unit_to_mw(self.gen["cost_pu"][gen_id]),
            )

        try:
            pp.rundcopp(
                self.grid_backend,
                verbose=verbose,
                suppress_warnings=True,
                delta=self.params.tol,
            )
            valid = True
        except pp.optimal_powerflow.OPFNotConverged as e:
            valid = False
            print(e)

        self._solve_save_backend()

        generators_p = self.grid_backend.res_gen["p_mw"].sum()
        ext_grids_p = self.grid_backend.res_ext_grid["p_mw"].sum()
        loads_p = self.grid_backend.load["p_mw"].sum()
        res_loads_p = self.grid_backend.res_load["p_mw"].sum()
        valid = (
            valid
            and np.abs(generators_p + ext_grids_p - loads_p) < 1e-2
            and np.abs(res_loads_p - loads_p) < 1e-2
        )

        result = {
            "res_cost": self.grid_backend.res_cost,
            "res_bus": self.grid_backend.res_bus,
            "res_line": self.grid_backend.res_line,
            "res_gen": self.grid_backend.res_gen,
            "res_load": self.grid_backend.res_load,
            "res_ext_grid": self.grid_backend.res_ext_grid,
            "res_trafo": self.grid_backend.res_trafo,
            "valid": valid,
        }
        return result

    def solve_and_compare(self, verbose=False):
        result = self.solve(verbose=False)
        result_backend = self.solve_backend(verbose=False)

        res_cost = pd.DataFrame(
            {
                "objective": [result["res_cost"]],
                "b_objective": [result_backend["res_cost"]],
                "diff": np.abs(result["res_cost"] - result_backend["res_cost"]),
            }
        )

        res_bus = pd.DataFrame(
            {
                "delta_pu": result["res_bus"]["delta_pu"],
                "b_delta_pu": result_backend["res_bus"]["delta_pu"],
                "diff": np.abs(
                    result["res_bus"]["delta_pu"]
                    - result_backend["res_bus"]["delta_pu"]
                ),
            }
        )

        res_line = pd.DataFrame(
            {
                "p_pu": result["res_line"]["p_pu"],
                "b_p_pu": result_backend["res_line"]["p_pu"],
                "diff": np.abs(
                    result["res_line"]["p_pu"] - result_backend["res_line"]["p_pu"]
                ),
                "line_loading": result["res_line"]["loading_percent"],
                "b_line_loading": result_backend["res_line"]["loading_percent"],
                "diff_loading": np.abs(
                    result["res_line"]["loading_percent"]
                    - result_backend["res_line"]["loading_percent"]
                ),
                "max_p_pu": np.abs(result["res_line"]["p_pu"])
                / result["res_line"]["loading_percent"],
                "b_max_p_pu": np.abs(result_backend["res_line"]["p_pu"])
                / result_backend["res_line"]["loading_percent"],
            }
        )
        res_line["diff_max"] = np.abs(res_line["max_p_pu"] - res_line["b_max_p_pu"])

        res_gen = pd.DataFrame(
            {
                "gen_pu": result["res_gen"]["p_pu"],
                "b_gen_pu": result_backend["res_gen"]["p_pu"],
                "diff": np.abs(
                    result["res_gen"]["p_pu"] - result_backend["res_gen"]["p_pu"]
                ),
                "gen_cost_pu": self.gen["cost_pu"],
            }
        )

        res_load = pd.DataFrame(
            {
                "load_pu": result["res_load"]["p_pu"],
                "b_load_pu": result_backend["res_load"]["p_pu"],
                "diff": np.abs(
                    result["res_load"]["p_pu"] - result_backend["res_load"]["p_pu"]
                ),
            }
        )

        res_ext_grid = pd.DataFrame(
            {
                "ext_grid_pu": result["res_ext_grid"]["p_pu"],
                "b_ext_grid_pu": result_backend["res_ext_grid"]["p_pu"],
                "diff": np.abs(
                    result["res_ext_grid"]["p_pu"]
                    - result_backend["res_ext_grid"]["p_pu"]
                ),
            }
        )

        res_trafo = pd.DataFrame(
            {
                "trafo_pu": result["res_trafo"]["p_pu"],
                "b_trafo_pu": result_backend["res_trafo"]["p_pu"],
                "diff": np.abs(
                    result["res_trafo"]["p_pu"] - result_backend["res_trafo"]["p_pu"]
                ),
                "trafo_loading": result["res_trafo"]["loading_percent"],
                "b_trafo_loading": result_backend["res_trafo"]["loading_percent"],
                "diff_loading": np.abs(
                    result["res_trafo"]["loading_percent"]
                    - result_backend["res_trafo"]["loading_percent"]
                ),
                "max_p_pu": np.abs(result["res_trafo"]["p_pu"])
                / result["res_trafo"]["loading_percent"],
                "b_max_p_pu": np.abs(result_backend["res_trafo"]["p_pu"])
                / result_backend["res_trafo"]["loading_percent"],
            }
        )
        res_trafo["diff_max"] = np.abs(res_trafo["max_p_pu"] - res_trafo["b_max_p_pu"])

        if verbose:
            print("OBJECTIVE\n" + res_cost.to_string())
            print("BUS\n" + res_bus.to_string())
            print("LINE\n" + res_line.to_string())
            print("GEN\n" + res_gen.to_string())
            print("LOAD\n" + res_load.to_string())
            if len(res_ext_grid.index):
                print("EXT GRID\n" + res_ext_grid.to_string())

            if len(res_trafo.index):
                print("TRAFO\n" + res_trafo.to_string())

        result = {
            "res_cost": res_cost,
            "res_bus": res_bus,
            "res_line": res_line,
            "res_gen": res_gen,
            "res_load": res_load,
            "res_ext_grid": res_ext_grid,
            "res_trafo": res_trafo,
        }
        return result

    """
        PRINT FUNCTIONS.
    """

    def print_results(self):
        print("\nRESULTS\n")
        print("{:<10}{}".format("OBJECTIVE", self.res_cost))
        print("RES BUS\n" + self.res_bus.to_string())
        print("RES LINE\n" + self.res_line.to_string())
        print("RES GEN\n" + self.res_gen.to_string())
        print("RES LOAD\n" + self.res_load.to_string())
        print("RES EXT GRID\n" + self.res_ext_grid.to_string())
        print("RES TRAFO\n" + self.res_trafo.to_string())

    def print_results_backend(self):
        res_cost = self.grid_backend.res_cost
        res_bus = self.grid_backend.res_bus[["delta_pu"]]
        res_line = self.grid_backend.res_line[["p_pu", "max_p_pu", "loading_percent"]]
        res_gen = self.grid_backend.res_gen[["min_p_pu", "p_pu", "max_p_pu", "cost_pu"]]
        res_load = self.grid_backend.res_load[["p_pu"]]
        res_ext_grid = self.grid_backend.res_ext_grid[["p_pu"]]
        res_trafo = self.grid_backend.res_trafo[["p_pu", "max_p_pu", "loading_percent"]]

        print("\nRESULTS BACKEND\n")
        print("{:<10}{}".format("OBJECTIVE", res_cost))
        print("RES BUS\n" + res_bus.to_string())
        print("RES LINE\n" + res_line.to_string())
        print("RES GEN\n" + res_gen.to_string())
        print("RES LOAD\n" + res_load.to_string())
        print("RES EXT GRID\n" + res_ext_grid.to_string())
        print("RES TRAFO\n" + res_trafo.to_string())

    def print_model(self):
        print(self.model.pprint())
