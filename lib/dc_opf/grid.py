import numpy as np
import pandas as pd

from .topology_converter import TopologyConverter, bus_names_to_sub_ids
from .unit_converter import UnitConverter
from ..data_utils import hot_to_indices
from ..visualizer import describe_substation


class GridDCOPF(UnitConverter, TopologyConverter):
    def __init__(self, case, base_unit_v, base_unit_p=1e6):
        UnitConverter.__init__(self, base_unit_v=base_unit_v, base_unit_p=base_unit_p)
        self.case = case

        if self.case.env:
            self.env_dc = self.case.env.parameters.ENV_DC

        # Initialize grid elements
        self.sub = pd.DataFrame(
            columns=[
                "id",
                "bus",
                "line_or",
                "line_ex",
                "gen",
                "load",
                "ext_grid",
                "n_elements",
                "cooldown",
            ]
        )
        self.bus = pd.DataFrame(
            columns=[
                "id",
                "sub",
                "sub_bus",
                "v_pu",
                "line_or",
                "line_ex",
                "gen",
                "load",
                "ext_grid",
                "n_elements",
            ]
        )
        self.line = pd.DataFrame(
            columns=[
                "id",
                "sub_or",
                "sub_ex",
                "bus_or",
                "bus_ex",
                "sub_bus_or",
                "sub_bus_ex",
                "b_pu",
                "p_pu",
                "max_p_pu",
                "max_p_pu_dc",
                "status",
                "trafo",
                "cooldown",
                "t_overflow",
            ]
        )
        self.gen = pd.DataFrame(
            columns=[
                "id",
                "sub",
                "bus",
                "sub_bus",
                "p_pu",
                "min_p_pu",
                "max_p_pu",
                "cost_pu",
            ]
        )
        self.load = pd.DataFrame(columns=["id", "sub", "bus", "sub_bus", "p_pu"])

        # External grid
        self.ext_grid = pd.DataFrame(
            columns=["id", "sub", "bus", "sub_bus", "p_pu", "min_p_pu", "max_p_pu"]
        )

        # Transformer
        self.trafo = pd.DataFrame(
            columns=[
                "id",
                "sub_or",
                "sub_ex",
                "bus_or",
                "bus_ex",
                "sub_bus_or",
                "sub_bus_ex",
                "b_pu",
                "p_pu",
                "max_p_pu",
                "max_p_pu_dc",
                "status",
                "trafo",
            ]
        )

        self.slack_bus = None
        self.fixed_elements = None
        self.max_rho = None

        self.build_grid()

        # If grid is part of grid2op environment, then construct a topology converter
        if self.case.env:
            TopologyConverter.__init__(self, env=case.env)
            self._is_valid_grid()
            self.topo_vect, self.line_status = self._get_topology_vector()

    def __str__(self):
        output = "Grid p. u.\n"
        output = (
            output + f"\t - Substations {self.sub.shape} {list(self.sub.columns)}\n"
        )
        output = output + f"\t - Buses {self.bus.shape} {list(self.bus.columns)}\n"
        output = (
            output
            + f"\t - Power lines {self.line[~self.line.trafo].shape} {list(self.line[~self.line.trafo].columns)}\n"
        )
        output = output + f"\t - Generators {self.gen.shape} {list(self.gen.columns)}\n"
        output = output + f"\t - Loads {self.load.shape} {list(self.load.columns)}\n"
        output = (
            output
            + f"\t - External grids {self.ext_grid.shape} {list(self.ext_grid.columns)}\n"
        )
        output = (
            output + f"\t - Transformers {self.trafo.shape} {list(self.trafo.columns)}"
        )
        return output

    def build_grid(self):
        """
        Buses.
        """
        self.bus["id"] = self.case.grid_backend.bus.index
        self.bus["sub"] = bus_names_to_sub_ids(self.case.grid_backend.bus["name"])
        self.bus["v_pu"] = self.convert_kv_to_per_unit(
            self.case.grid_backend.bus["vn_kv"]
        )

        """
            Substations.
        """
        self.sub["id"] = sorted(self.bus["sub"].unique())

        """
            Power lines.
        """
        self.line["id"] = self.case.grid_backend.line.index

        # Inverse line reactance
        # Equation given: https://pandapower.readthedocs.io/en/v2.2.2/elements/line.html.

        x_pu = self.convert_ohm_to_per_unit(
            self.case.grid_backend.line["x_ohm_per_km"]
            * self.case.grid_backend.line["length_km"]
            / self.case.grid_backend.line["parallel"]
        )
        self.line["b_pu"] = 1 / x_pu



        if self.case.name == "Case L2RPN 2020 WCCI":
            self.line["b_pu"][[45, 46, 47]] = [5000.0, 1014.19878296, 3311.25827815]

        # Power line flow thermal limit
        # P_l_max = I_l_max * V_l
        line_max_i_pu = self.convert_ka_to_per_unit(
            self.case.grid_backend.line["max_i_ka"]
        )
        self.line["max_p_pu"] = (
            np.sqrt(3)
            * line_max_i_pu
            * self.bus["v_pu"].values[self.case.grid_backend.line["from_bus"].values]
        )
        self.line["max_p_pu_dc"] = self.line["max_p_pu"].values

        self.line["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.res_line["p_from_mw"]
        )

        # Line status
        self.line["status"] = self.case.grid_backend.line["in_service"]

        """
            Generators.
        """
        self.gen["id"] = self.case.grid_backend.gen.index
        self.gen["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.gen["p_mw"]
        )
        self.gen["max_p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.gen["max_p_mw"]
        )
        self.gen["min_p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.gen["min_p_mw"]
        )

        self.gen["min_p_pu"] = self.gen["min_p_pu"].values
        self.gen["cost_pu"] = 1.0

        """
            Loads.
        """
        self.load["id"] = self.case.grid_backend.load.index
        self.load["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.load["p_mw"]
        )

        """
            External grids.
        """
        self.ext_grid["id"] = self.case.grid_backend.ext_grid.index
        self.ext_grid["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.res_ext_grid["p_mw"]
        )
        if "min_p_mw" in self.case.grid_backend.ext_grid.columns:
            self.ext_grid["min_p_pu"] = self.convert_mw_to_per_unit(
                self.case.grid_backend.ext_grid["min_p_mw"]
            )

        if "max_p_mw" in self.case.grid_backend.ext_grid.columns:
            self.ext_grid["max_p_pu"] = self.convert_mw_to_per_unit(
                self.case.grid_backend.ext_grid["max_p_mw"]
            )

        """
            Transformers.
            "High voltage bus is the origin (or) bus."
            Follows definitions from https://pandapower.readthedocs.io/en/v2.2.2/elements/trafo.html.
        """

        self.trafo["id"] = self.case.grid_backend.trafo.index
        if "b_pu" in self.case.grid_backend.trafo.columns:
            self.trafo["b_pu"] = self.case.grid_backend.trafo["b_pu"]
        else:
            self.trafo["b_pu"] = (
                1
                / (self.case.grid_backend.trafo["vk_percent"] / 100.0)
                * self.case.grid_backend.trafo["sn_mva"]
            )

        self.trafo["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.res_trafo["p_hv_mw"]
        )
        self.trafo["p_pu"].fillna(0, inplace=True)

        if "max_p_pu" in self.case.grid_backend.trafo.columns:
            self.trafo["max_p_pu"] = self.case.grid_backend.trafo["max_p_pu"]
        else:
            self.trafo["max_p_pu"] = self.convert_mw_to_per_unit(
                self.case.grid_backend.trafo["sn_mva"]
            )
        self.trafo["max_p_pu_dc"] = self.trafo["max_p_pu"].values

        self.trafo["status"] = self.case.grid_backend.trafo["in_service"]

        # Reindex
        self.sub.set_index("id", inplace=True)
        self.bus.set_index("id", inplace=True)
        self.line.set_index("id", inplace=True)
        self.gen.set_index("id", inplace=True)
        self.load.set_index("id", inplace=True)
        self.ext_grid.set_index("id", inplace=True)
        self.trafo.set_index("id", inplace=True)

        """
            Topology.
        """
        # Generators
        self.gen["bus"] = self.case.grid_backend.gen["bus"]
        self.gen["sub"] = self.sub.index.values[self.gen["bus"]]

        # Loads
        self.load["bus"] = self.case.grid_backend.load["bus"]
        self.load["sub"] = self.sub.index.values[self.load["bus"]]

        # Power lines
        self.line["bus_or"] = self.case.grid_backend.line["from_bus"]
        self.line["bus_ex"] = self.case.grid_backend.line["to_bus"]
        self.line["sub_or"] = self.sub.index.values[self.line["bus_or"]]
        self.line["sub_ex"] = self.sub.index.values[self.line["bus_ex"]]

        # External grids
        self.ext_grid["bus"] = self.case.grid_backend.ext_grid["bus"]
        self.ext_grid["sub"] = self.sub.index.values[self.ext_grid["bus"]]

        # Transformers
        self.trafo["bus_or"] = self.case.grid_backend.trafo["hv_bus"]
        self.trafo["bus_ex"] = self.case.grid_backend.trafo["lv_bus"]
        self.trafo["sub_or"] = self.sub.index.values[self.trafo["bus_or"]]
        self.trafo["sub_ex"] = self.sub.index.values[self.trafo["bus_ex"]]

        # Merge power lines and transformers
        self.line["trafo"] = False
        self.trafo["trafo"] = True
        self.line = self.line.append(self.trafo, ignore_index=True)

        sub_bus = np.empty_like(self.bus.index)
        for sub_id in self.sub.index:
            bus_mask = self.bus["sub"] == sub_id
            gen_mask = self.gen["sub"] == sub_id
            load_mask = self.load["sub"] == sub_id
            line_or_mask = self.line["sub_or"] == sub_id
            line_ex_mask = self.line["sub_ex"] == sub_id
            ext_grid_mask = self.ext_grid["sub"] == sub_id

            sub_bus[bus_mask] = np.arange(1, np.sum(bus_mask) + 1)

            self.sub["bus"][sub_id] = tuple(np.flatnonzero(bus_mask))
            self.sub["gen"][sub_id] = tuple(np.flatnonzero(gen_mask))
            self.sub["load"][sub_id] = tuple(np.flatnonzero(load_mask))
            self.sub["line_or"][sub_id] = tuple(np.flatnonzero(line_or_mask))
            self.sub["line_ex"][sub_id] = tuple(np.flatnonzero(line_ex_mask))
            self.sub["ext_grid"][sub_id] = tuple(np.flatnonzero(ext_grid_mask))

        # Bus within a substation of each grid element
        self.bus["sub_bus"] = sub_bus
        self.gen["sub_bus"] = self.bus["sub_bus"].values[self.gen["bus"].values]
        self.load["sub_bus"] = self.bus["sub_bus"].values[self.load["bus"].values]
        self.line["sub_bus_or"] = self.bus["sub_bus"].values[self.line["bus_or"].values]
        self.line["sub_bus_ex"] = self.bus["sub_bus"].values[self.line["bus_ex"].values]
        self.ext_grid["sub_bus"] = self.bus["sub_bus"].values[
            self.ext_grid["bus"].values
        ]
        self.trafo["sub_bus_or"] = self.bus["sub_bus"].values[
            self.trafo["bus_or"].values
        ]
        self.trafo["sub_bus_ex"] = self.bus["sub_bus"].values[
            self.trafo["bus_ex"].values
        ]

        self._update_buses()

        # Number of elements per substation (without external grids)
        if self.case.env:
            self.sub["n_elements"] = self.case.env.sub_info
        else:
            self.sub["n_elements"] = [
                len(self.sub.line_or[sub_id])
                + len(self.sub.line_ex[sub_id])
                + len(self.sub.gen[sub_id])
                + len(self.sub.load[sub_id])
                for sub_id in self.sub.index
            ]

        self.bus["n_elements"] = [
            len(self.bus.line_or[bus_id])
            + len(self.bus.line_ex[bus_id])
            + len(self.bus.gen[bus_id])
            + len(self.bus.load[bus_id])
            for bus_id in self.bus.index
        ]

        # Cooldown
        self.sub["cooldown"] = 0
        self.line["cooldown"] = 0

        # Overflow
        self.line["t_overflow"] = 0

        # Maintenance
        self.line["next_maintenance"] = -1
        self.line["duration_maintenance"] = 0

        # Fill with 0 if no value
        self.line["p_pu"] = self.line["p_pu"].fillna(0)
        self.gen["p_pu"] = self.gen["p_pu"].fillna(0)
        self.ext_grid["p_pu"] = self.ext_grid["p_pu"].fillna(0)

        # Grid and computation parameters
        if not self.case.grid_backend.gen["slack"].any():
            slack_bus = 0
            if len(self.ext_grid.index):
                slack_bus = self.case.grid_backend.ext_grid["bus"][0]
            self.slack_bus = slack_bus
        else:
            self.slack_bus = self.gen.bus[
                np.flatnonzero(self.case.grid_backend.gen["slack"])[0]
            ]

        # Substation topological symmetry
        self.fixed_elements = self.get_fixed_elements()

        # Big-M for power flows
        self.max_rho = 2.0

    def print_grid(self):
        print("\nGRID\n")
        print("SUB\n" + self.sub.to_string())
        print("BUS\n" + self.bus.to_string())
        print("LINE\n" + self.line[~self.line["trafo"]].to_string())
        print("GEN\n" + self.gen.to_string())
        print("LOAD\n" + self.load.to_string())
        if len(self.ext_grid.index):
            print("EXT GRID\n" + self.ext_grid.to_string())
        if len(self.trafo.index):
            print("TRAFO\n" + self.trafo.to_string())
        print(f"SLACK BUS: {self.slack_bus}")
        print("FIXED ELEMENTS\n" + self.fixed_elements.to_string())

    def get_fixed_elements(self, verbose=False):
        """
        Get id of a power line end at each substation. Used for eliminating substation topological symmetry.
        """
        fixed_elements = dict()

        for sub_id in self.sub.index:
            fixed_elements[sub_id] = dict()

            if self.case.env:
                # Grid element ids
                line_or_ids = hot_to_indices(
                    self.case.env.action_space.line_or_to_subid == sub_id
                )
                line_ex_ids = hot_to_indices(
                    self.case.env.action_space.line_ex_to_subid == sub_id
                )

                # Grid element positions within substation
                lines_or_pos = self.case.env.action_space.line_or_to_sub_pos[
                    line_or_ids
                ]
                lines_ex_pos = self.case.env.action_space.line_ex_to_sub_pos[
                    line_ex_ids
                ]

                fixed_elements[sub_id]["line_or"] = line_or_ids[
                    np.flatnonzero(lines_or_pos == 0)
                ].tolist()
                fixed_elements[sub_id]["line_ex"] = line_ex_ids[
                    np.flatnonzero(lines_ex_pos == 0)
                ].tolist()

                if verbose:
                    describe_substation(sub_id, self.case.env)
            else:
                line_or_ids = self.sub["line_or"][sub_id]
                line_ex_ids = self.sub["line_ex"][sub_id]
                if len(line_or_ids):
                    fixed_elements[sub_id]["line_or"] = [line_or_ids[0]]
                    fixed_elements[sub_id]["line_ex"] = []
                elif len(line_ex_ids):
                    fixed_elements[sub_id]["line_or"] = []
                    fixed_elements[sub_id]["line_ex"] = [line_ex_ids[0]]

            # Check if each substation has exactly one power line end at position 0
            assert (
                len(fixed_elements[sub_id]["line_or"])
                + len(fixed_elements[sub_id]["line_ex"])
                == 1
            )

            if verbose:
                print(fixed_elements[sub_id])

        fixed_elements = pd.DataFrame(
            [fixed_elements[sub_id] for sub_id in fixed_elements]
        )
        return fixed_elements

    """
        UPDATE GRID GIVEN AN OBSERVATION.
    """

    def update(self, obs_new, reset=False, verbose=False):
        self._update_grid_topology(obs_new, reset=reset, verbose=verbose)
        self._update_flows(obs_new)

        self._count_topology_changes(
            self.topo_vect,
            self.line_status,
            obs_new.topo_vect,
            obs_new.line_status,
            verbose=verbose,
        )

        self.topo_vect, self.line_status = self._get_topology_vector()

        if not reset and not self.env_dc:
            self._update_ac_thermal_limits(obs_new, verbose=verbose)

        # Check if topology vector and line statuses are updated
        assert np.equal(self.topo_vect, obs_new.topo_vect).all()
        assert np.equal(self.line_status, obs_new.line_status).all()

    def _update_injection_topology(
        self, injections, inj_sub_bus, inj_str, reset=False, verbose=False
    ):
        for inj_id in injections.index:
            sub_bus = injections["sub_bus"][inj_id]
            bus = injections["bus"][inj_id]

            sub_bus_new = inj_sub_bus[inj_id]
            bus_new = self.sub["bus"][injections["sub"][inj_id]][sub_bus_new - 1]

            if verbose and bus != bus_new:
                print(
                    "{:<35}{:<10}".format(
                        f"{inj_str} {inj_id}:",
                        f"{bus}({sub_bus})\t->\t{bus_new}({sub_bus_new})",
                    )
                )

            if not reset:
                assert (bus != bus_new) == (
                    sub_bus != sub_bus_new
                )  # If switch, then both have to change
            if bus != bus_new and sub_bus != sub_bus_new:
                injections["sub_bus"][inj_id] = sub_bus_new
                injections["bus"][inj_id] = bus_new

        assert np.equal(injections["sub_bus"], inj_sub_bus).all()

    def _update_grid_topology(
        self,
        obs_new,
        reset=False,
        verbose=False,
    ):
        """
        Update grid given an observation.
        """
        (
            gen_sub_bus,
            load_sub_bus,
            line_or_sub_bus,
            line_ex_sub_bus,
        ) = self._get_substation_buses(obs_new.topo_vect)
        line_status = obs_new.line_status

        # Update generators
        self._update_injection_topology(
            self.gen, gen_sub_bus, "GEN", reset=reset, verbose=verbose
        )

        # Update loads
        self._update_injection_topology(
            self.load, load_sub_bus, "LOAD", reset=reset, verbose=verbose
        )

        # Update power lines
        for line_id in self.line.index:
            status = self.line["status"][line_id]
            sub_bus_or = self.line["sub_bus_or"][line_id]
            bus_or = self.line["bus_or"][line_id]
            sub_bus_ex = self.line["sub_bus_ex"][line_id]
            bus_ex = self.line["bus_ex"][line_id]

            status_new = line_status[line_id]

            if status_new:
                sub_bus_or_new = line_or_sub_bus[line_id]
                bus_or_new = self.sub["bus"][self.line["sub_or"][line_id]][
                    sub_bus_or_new - 1
                ]
                sub_bus_ex_new = line_ex_sub_bus[line_id]
                bus_ex_new = self.sub["bus"][self.line["sub_ex"][line_id]][
                    sub_bus_ex_new - 1
                ]

                if not reset:
                    assert (bus_or != bus_or_new) == (
                        sub_bus_or != sub_bus_or_new
                    )  # If switch, then both have to change
                    assert (bus_ex != bus_ex_new) == (
                        sub_bus_ex != sub_bus_ex_new
                    )  # If switch, then both have to chang

                if bus_or != bus_or_new and sub_bus_or != sub_bus_or_new:
                    if verbose:
                        print(
                            "{:<35}{:<10}".format(
                                f"LINE OR {line_id}:",
                                f"{bus_or}({sub_bus_or})\t->\t{bus_or_new}({sub_bus_or_new})",
                            )
                        )
                    self.line["sub_bus_or"][line_id] = sub_bus_or_new
                    self.line["bus_or"][line_id] = bus_or_new
                if bus_ex != bus_ex_new and sub_bus_ex != sub_bus_ex_new:
                    if verbose:
                        print(
                            "{:<35}{:<10}".format(
                                f"LINE EX {line_id}:",
                                f"{bus_ex}({sub_bus_ex})\t->\t{bus_ex_new}({sub_bus_ex_new})",
                            )
                        )
                    self.line["sub_bus_ex"][line_id] = sub_bus_ex_new
                    self.line["bus_ex"][line_id] = bus_ex_new

            # In case of status switch
            if status != status_new:
                self.line["status"][line_id] = status_new

                if verbose:
                    print(
                        "{:<35}{:<10}".format(
                            f"LINE STATUS {line_id}:", f"{status}  ->  {status_new}"
                        )
                    )

        # Update transformers
        self.trafo["bus_or"] = self.line["bus_or"].values[self.line["trafo"]]
        self.trafo["sub_bus_or"] = self.line["sub_bus_or"].values[self.line["trafo"]]
        self.trafo["bus_ex"] = self.line["bus_ex"].values[self.line["trafo"]]
        self.trafo["sub_bus_ex"] = self.line["sub_bus_ex"].values[self.line["trafo"]]

        # Cooldown
        # if time = 0, then action is legal
        self.sub["cooldown"] = obs_new.time_before_cooldown_sub
        self.line["cooldown"] = obs_new.time_before_cooldown_line

        # Overflow
        self.line["t_overflow"] = obs_new.timestep_overflow

        # Maintenance
        self.line["next_maintenance"] = obs_new.time_next_maintenance
        self.line["duration_maintenance"] = obs_new.duration_next_maintenance

        if not reset:
            assert np.equal(
                self.line["sub_bus_or"][line_status], line_or_sub_bus[line_status]
            ).all()
            assert np.equal(
                self.line["sub_bus_ex"][line_status], line_ex_sub_bus[line_status]
            ).all()
            assert np.equal(self.line["status"], line_status).all()

        # Update grid.bus data
        self._update_buses()

    def _update_ac_thermal_limits(self, obs_new, verbose=False):
        v_pu = self.convert_kv_to_per_unit(obs_new.v_or)
        q_pu = self.convert_mw_to_per_unit(obs_new.q_or)

        max_i_pu = self.convert_a_to_per_unit(self.case.env.get_thermal_limit())
        max_s_pu = np.sqrt(3) * max_i_pu * v_pu

        rho = obs_new.rho

        self.line["max_p_pu"] = np.sqrt(
            np.maximum(np.square(max_s_pu) - np.square(q_pu / (rho + 1e-9)), 0.0)
        )

        res = self.line[["max_p_pu", "max_p_pu_dc"]].copy()
        res["env_max_p_pu"] = self.convert_mw_to_per_unit(
            np.divide(obs_new.p_or, rho + 1e-9)
        )

        if verbose:
            print(res)

    def _update_flows(self, obs):
        """
        Update active power flows, productions and demands.
        """
        # prod_p_f, _, load_p_f, _ = obs.get_forecasted_inj()
        prod_p_f = obs.prod_p
        load_p_f = obs.load_p

        self.gen["p_pu"] = self.convert_mw_to_per_unit(prod_p_f)
        self.load["p_pu"] = self.convert_mw_to_per_unit(load_p_f)
        self.line["p_pu"] = self.convert_mw_to_per_unit(obs.p_or)
        self.trafo["p_pu"] = self.convert_mw_to_per_unit(
            obs.p_or[self.line["trafo"].values]
        )

    def _update_buses(self):
        for bus_id in self.bus.index:
            gen_mask = self.gen["bus"] == bus_id
            load_mask = self.load["bus"] == bus_id
            line_or_mask = self.line["bus_or"] == bus_id
            line_ex_mask = self.line["bus_ex"] == bus_id
            ext_grid_mask = self.ext_grid["bus"] == bus_id

            self.bus["gen"][bus_id] = tuple(np.flatnonzero(gen_mask))
            self.bus["load"][bus_id] = tuple(np.flatnonzero(load_mask))
            self.bus["line_or"][bus_id] = tuple(np.flatnonzero(line_or_mask))
            self.bus["line_ex"][bus_id] = tuple(np.flatnonzero(line_ex_mask))
            self.bus["ext_grid"][bus_id] = tuple(np.flatnonzero(ext_grid_mask))

    def _get_topology_vector(self):
        gen_sub_bus = self.gen.sub_bus.values.copy()
        load_sub_bus = self.load.sub_bus.values.copy()
        line_or_sub_bus = self.line.sub_bus_or.values.copy()
        line_ex_sub_bus = self.line.sub_bus_ex.values.copy()
        line_status = self.line.status.values.copy()

        topo_vect = self._construct_topology_vector(
            gen_sub_bus, load_sub_bus, line_or_sub_bus, line_ex_sub_bus, line_status
        )

        return topo_vect, line_status

    def _is_valid_grid(self):
        assert self.n_sub == len(self.sub.index)
        assert self.n_bus == len(self.bus.index)
        assert self.n_gen == len(self.gen.index)
        assert self.n_load == len(self.load.index)
        assert self.n_line == len(self.line.index)

        for sub_id, n_elements_sub in enumerate(self.env.sub_info):
            (
                gen_ids,
                load_ids,
                line_or_ids,
                line_ex_ids,
            ) = self._get_substation_element_ids(sub_id)

            assert np.equal(gen_ids, self.sub.gen[sub_id]).all()
            assert np.equal(load_ids, self.sub.load[sub_id]).all()
            assert np.equal(line_or_ids, self.sub.line_or[sub_id]).all()
            assert np.equal(line_ex_ids, self.sub.line_ex[sub_id]).all()
            assert np.equal(self.env.sub_info, self.sub.n_elements).all()
