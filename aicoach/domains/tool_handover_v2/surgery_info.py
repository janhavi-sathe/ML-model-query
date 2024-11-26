from .define import Requirement, PatientVital, NurseDirection

CABG_INFO = {
    "width":
    7,
    "height":
    7,
    "surgeon_pos": (3, 3),
    "patient_pos_size": (1, 4, 4, 1),
    "perf_pos": (3, 5),
    "anes_pos": (4, 5),
    "nurse_init_pos": (2, 3),
    "nurse_init_dir":
    NurseDirection.Down,
    "nurse_possible_pos": [(2, 3), (2, 2), (2, 1)],
    "table_blocks": [(1, 1), (1, 2), (1, 3)],
    "vital_pos": (5, 4),
    "surgical_steps": [
        {
            "name": "Sterile Prepping",
            PatientVital.Stable: [(Requirement.Antiseptic_Solution, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Pre-incision Time out",
            PatientVital.Stable: [(Requirement.Hand_Only, 1)],
            PatientVital.Unstable: [(Requirement.Lab_Test, 1)],
        },
        {
            "name": "Sternotomy 1/2",
            PatientVital.Stable: [(Requirement.Scalpel, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Sternotomy 2/2",
            PatientVital.Stable: [(Requirement.Sternal_Saw, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Vessel Harvesting",
            PatientVital.Stable: [(Requirement.EVH_system, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Heparinization",
            PatientVital.Stable: [(Requirement.Hand_Only, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Cannulation 1/2",
            PatientVital.Stable: [(Requirement.Tourniquet, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Cannulation 2/2",
            PatientVital.Stable: [(Requirement.Needle, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Initiate CPB",
            PatientVital.Stable: [(Requirement.Dressing, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Clamp Aorta and Initiate Cardioplegia",
            PatientVital.Stable: [(Requirement.Clamp, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Anastomoses",
            PatientVital.Stable: [(Requirement.Forceps, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Separate from Bypass",
            PatientVital.Stable: [(Requirement.Hand_Only, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
        {
            "name": "Sternal Closure",
            PatientVital.Stable: [(Requirement.Suture, 1)],
            PatientVital.Unstable: [(Requirement.Nurse_Assist, 1)],
        },
    ],
    "tool_table_zone": {
        Requirement.Antiseptic_Solution: (0, 1),  # (block idx, quadrant)
        Requirement.Lab_Test: (0, 2),
        Requirement.Scalpel: (0, 3),
        Requirement.Sternal_Saw: (0, 4),
        Requirement.Clamp: (1, 1),
        Requirement.Needle: (1, 2),
        Requirement.EVH_system: (1, 3),
        Requirement.Tourniquet: (1, 4),
        Requirement.Dressing: (2, 1),
        Requirement.Forceps: (2, 2),
        Requirement.Suture: (2, 3)
    }
}
