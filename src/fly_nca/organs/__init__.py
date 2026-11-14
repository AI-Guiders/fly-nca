"""L3 Topological NCA Organs of Drosophila melanogaster."""

from fly_nca.organs.antennal_sez import AntennalSEZNCA, ChemosensoryDigest
from fly_nca.organs.central_complex import CentralComplexNCA, CXNavigationReadout
from fly_nca.organs.lh_spin_gate import LHSpinGateNCA, LHSpinOutput
from fly_nca.organs.mushroom_body import (
    COMP_GAMMA1,
    COMP_GAMMA5,
    N_MB_COMPARTMENTS,
    MBStepOutput,
    MushroomBodyNCA,
)
from fly_nca.organs.optic_lobe import OpticLobeNCA, VisualProjectionDigest
from fly_nca.organs.vnc_autopilot import (
    N_VNC_CELLS,
    DescendingJoystickDN,
    VNCAutopilotNCA,
)

__all__ = [
    "OpticLobeNCA",
    "VisualProjectionDigest",
    "AntennalSEZNCA",
    "ChemosensoryDigest",
    "LHSpinGateNCA",
    "LHSpinOutput",
    "N_MB_COMPARTMENTS",
    "COMP_GAMMA1",
    "COMP_GAMMA5",
    "MBStepOutput",
    "MushroomBodyNCA",
    "CentralComplexNCA",
    "CXNavigationReadout",
    "N_VNC_CELLS",
    "DescendingJoystickDN",
    "VNCAutopilotNCA",
]
