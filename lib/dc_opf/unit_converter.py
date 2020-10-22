import numpy as np

from ..visualizer import pprint


class UnitConverter:
    def __init__(self, base_unit_p=1e6, base_unit_v=1e5):
        # Base units
        self.base_unit_p = base_unit_p  # Base unit is 1 MVA or 1 MW or 1 MVar
        self.base_unit_v = base_unit_v  # Base unit is 100 kV
        self.base_unit_i = self.base_unit_p / self.base_unit_v  # Base unit is 1 A
        self.base_unit_z = self.base_unit_v ** 2 / self.base_unit_p

        self.unit_p_mw = 1e6
        self.unit_a_ka = 1e3
        self.unit_v_kv = 1e3

    def print_base_units(self):
        pprint("Quantity", "Value")
        pprint("    - P:", "{} W".format(self.base_unit_p))
        pprint("    - U:", "{} V".format(self.base_unit_v))
        pprint("    - I:", "{} A".format(self.base_unit_i))
        pprint("    - Z:", "{} Ohm".format(self.base_unit_z))

    def convert_mw_to_per_unit(self, p_mw):
        p_pu = p_mw * self.unit_p_mw / self.base_unit_p
        return p_pu

    def convert_ohm_to_per_unit(self, z_ohm):
        z_pu = z_ohm / self.base_unit_z
        return z_pu

    def convert_a_to_per_unit(self, i_a):
        i_pu = i_a / self.base_unit_i
        return i_pu

    def convert_ka_to_per_unit(self, i_ka):
        i_pu = i_ka * self.unit_a_ka / self.base_unit_i
        return i_pu

    def convert_kv_to_per_unit(self, v_kv):
        v_pu = v_kv * self.unit_v_kv / self.base_unit_v
        return v_pu

    def convert_per_unit_to_mw(self, p_pu):
        p_mw = p_pu * self.base_unit_p / self.unit_p_mw
        return p_mw

    def convert_per_unit_to_ka(self, i_pu):
        i_ka = i_pu * self.base_unit_i / self.unit_a_ka
        return i_ka

    @staticmethod
    def convert_degree_to_rad(deg):
        rad = deg / 180.0 * np.pi
        return rad

    @staticmethod
    def convert_rad_to_deg(rad):
        deg = rad / np.pi * 180.0
        return deg
