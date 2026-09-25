"""Class definitions for SISVSE and EndoVis18, and the harmonized instrument-part taxonomy.

SISVSE labels individual robotic tools by part (head/wrist/body); EndoVis18 labels
generic instrument parts (clasper/wrist/shaft). Both map onto HARMONIZED = {tip, wrist, shaft},
which is what cross-dataset evaluation uses.
"""

import numpy as np

SISVSE_CLASSES = [
    "Background",
    "HarmonicAce_Head", "HarmonicAce_Body",
    "MarylandBipolarForceps_Head", "MarylandBipolarForceps_Wrist", "MarylandBipolarForceps_Body",
    "CadiereForceps_Head", "CadiereForceps_Wrist", "CadiereForceps_Body",
    "CurvedAtraumaticGrasper_Head", "CurvedAtraumaticGrasper_Body",
    "Stapler_Head", "Stapler_Body",
    "MediumLargeClipApplier_Head", "MediumLargeClipApplier_Wrist", "MediumLargeClipApplier_Body",
    "SmallClipApplier_Head", "SmallClipApplier_Wrist", "SmallClipApplier_Body",
    "SuctionIrrigation", "Needle", "Endotip", "Specimenbag", "DrainTube",
    "Liver", "Stomach", "Pancreas", "Spleen", "Gallbladder",
    "Gauze", "TheOther_Instruments", "TheOther_Tissues",
]
assert len(SISVSE_CLASSES) == 32

ENDOVIS18_CLASSES = [
    "background-tissue", "instrument-shaft", "instrument-clasper", "instrument-wrist",
    "kidney-parenchyma", "covered-kidney", "thread", "clamps", "suturing-needle",
    "suction-instrument", "small-intestine", "ultrasound-probe",
]

HARMONIZED = ["other", "tip", "wrist", "shaft"]
ANATOMY = ["Liver", "Stomach", "Pancreas", "Spleen", "Gallbladder"]


def _sisvse_to_harmonized() -> np.ndarray:
    lut = np.zeros(256, np.uint8)
    for i, name in enumerate(SISVSE_CLASSES):
        if name.endswith("_Head"):
            lut[i] = 1
        elif name.endswith("_Wrist"):
            lut[i] = 2
        elif name.endswith("_Body"):
            lut[i] = 3
    return lut


def _endovis18_to_harmonized() -> np.ndarray:
    lut = np.zeros(256, np.uint8)
    lut[ENDOVIS18_CLASSES.index("instrument-clasper")] = 1
    lut[ENDOVIS18_CLASSES.index("instrument-wrist")] = 2
    lut[ENDOVIS18_CLASSES.index("instrument-shaft")] = 3
    return lut


SISVSE_TO_HARMONIZED = _sisvse_to_harmonized()
ENDOVIS18_TO_HARMONIZED = _endovis18_to_harmonized()
