import numpy as np

def get_eye_event_columns():

    base = [
        "eye_tracker",                            
        "experiment_id",
        "participant_id",
        "filename",
        "trial_id",
        "stimuli_module",
        "stimuli_name",
        "timestamp",
        "duration",
        "x0",
        "y0",
        "x1",
        "y1",
        "token",
        "pupil",
        "amplitude",
        "peak_velocity",
        "eye_event_type"
    ]

    return base

def eye_event_list(eye_tracker: str, experiment_id: str,
                    participant_id: str, filename: str, trial_id: str, stimuli_module: str,
                    stimuli_name: str, timestamp: int, 
                    duration: float = np.nan, x0: float = np.nan, y0: float = np.nan,
                    x1: float = np.nan, y1: float = np.nan, token: list = [], pupil: int = np.nan, 
                    amplitude: float = np.nan, peak_velocity: float = np.nan, 
                    eye_event_type: str = "blink"):

    return  [
        eye_tracker,                           
        experiment_id,
        participant_id,
        filename,
        trial_id,
        stimuli_module,
        stimuli_name,
        timestamp,
        duration,
        x0,
        y0,
        x1,
        y1,
        token,
        pupil,
        amplitude,
        peak_velocity,
        eye_event_type,
    ]