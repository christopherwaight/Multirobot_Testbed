"""Repository paths, built from this file's location."""
import os
import sys

ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(ANALYSIS)
VF = os.path.join(ROOT, "trunk", "Python_Simulations", "Vector_Fields", "VF_Robot")
STUDY = os.path.join(VF, "experiments", "hyperparameter_study")
REVISION = os.path.join(VF, "experiments", "revision_metrics")
HW = os.path.join(ROOT, "New_Hardware_Data")
HW2026 = os.path.join(HW, "data_hardware_campaign_11_11_2026")
HW2025 = os.path.join(HW, "data_hardware_campaign_07_31_2025")
HW2025OCT = os.path.join(HW, "data_hardware_campaign_10_09_2025")
EXTRAS = os.path.join(ROOT, "Extras")
PAPER = os.path.join(ROOT, "Paper_Writing", "Systems Paper Revision", "cwaight_systems_paper")
OUT_HW = os.path.join(ANALYSIS, "hardware")
OUT_SIM = os.path.join(ANALYSIS, "simulation")
OUT_LC = os.path.join(ANALYSIS, "limit_cycle")
OUT_PAPER = os.path.join(ANALYSIS, "paper")

for p in (VF, STUDY, REVISION):
    if p not in sys.path:
        sys.path.insert(0, p)
