import numpy as np

def get_samples_columns(token: list = []):

    base = [
        "eye_tracker",                            
        "experiment_id",
        "participant_id",
        "filename",
        "trial_id",
        "stimuli_module",
        "stimuli_name",
    ]

    base.extend(token)
    return base

def samples_list(eye_tracker: str, experiment_id: str,
                    participant_id: str, filename: str, trial_id: str, stimuli_module: str,
                    stimuli_name: str, token: list = []):

    base = [
        eye_tracker,                           
        experiment_id,
        participant_id,
        filename,
        trial_id,
        stimuli_module,
        stimuli_name,
    ]

    base.extend(token)

    return base