import warnings

from .src import *

warnings.warn(
    "The d3il_lite.d3il_sim.sims.sl.teleoperation package is deprecated. Please use d3il_lite.d3il_sim.sims.sl.multibot_teleop instead",
    DeprecationWarning,
)
