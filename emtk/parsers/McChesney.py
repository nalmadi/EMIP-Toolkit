import os
import pandas as pd
import numpy as np
from .eye_events import get_eye_event_columns

EYE_TRACKER = "Tobii X3-120"
RAWDATA_MODULE = "emtk/datasets/McChesney2021/sessions"
STIMULI_MODULE = "emtk/datasets/McChesney2021/stimuli"

STIMULI_NAMES = (
    "P1Sa", "P1Sb"
    "P1Ca", "P1Cb",
    "P2Sa", "P2Sb"
    "P2Ca", "P2Cb",
    "P3Sa", "P3Sb"
    "P3Ca", "P3Cb",
)

STIMULIS_PER_EXPERIMENT = 3


def McChesney(sample_size: int = 216):
    """Import the McChesney2021 dataset.
    Fixation and saccades 

    Parameters
    ----------
    sample_size : int, optional (default 216)
        Number of subjects to be processed.

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe of eye events from every experiment in the dataset.
    """
    eye_events = pd.DataFrame()
    samples = pd.DataFrame()

    for r, _, f in os.walk(RAWDATA_MODULE):
        f.sort()
        for file in f:

            raw_data = pd.read_csv(
                os.path.join(r, file), sep="\t")

            # Delete AOI hit test (from column 52 to the end)
            raw_data.drop(
                raw_data.columns[52:], axis=1, inplace=True)

            # Extract data from the three trials only
            stimuli_start_end_idx = raw_data.loc[raw_data["Event value"].isin(
                STIMULI_NAMES)].index

            # experiment_samples = pd.DataFrame()
            # experiment_eye_events = pd.DataFrame()
            for idx in range(STIMULIS_PER_EXPERIMENT):
                trial_start = stimuli_start_end_idx[idx * 2]
                trial_end = stimuli_start_end_idx[idx * 2 + 1]
                trial_samples = raw_data.iloc[trial_start: trial_end].copy()

                experiment_id = file.split(".")[0].split("/")[-1]
                trial_id = idx + 1
                stimuli_name = "0{}_{}.png".format(
                    idx + 1, raw_data.at[trial_start, "Event value"])

                # Structure samples dataframe
                trial_samples["eye_tracker"] = EYE_TRACKER
                trial_samples["experiment_id"] = experiment_id
                trial_samples["participant_id"] = experiment_id
                trial_samples["filename"] = file
                trial_samples["trial_id"] = str(trial_id)
                trial_samples["stimuli_module"] = STIMULI_MODULE
                trial_samples["stimuli_name"] = stimuli_name

                samples = pd.concat([samples, trial_samples])

                # Extract fixation data to create eye events dataframe
                trial_eye_events = trial_samples[["Recording timestamp [ms]", "Gaze event duration [ms]",
                                                  "Gaze point X [DACS px]", "Gaze point Y [DACS px]",
                                                  "Eye movement type"]].copy()

                trial_eye_events.rename(columns={
                    "Recording timestamp [ms]": "timestamp",
                    "Gaze event duration [ms]": "duration",
                    "Gaze point X [DACS px]": "x0",
                    "Gaze point Y [DACS px]": "y0",
                    "Eye movement type": "eye_event_type",
                }, inplace=True)

                # Remove duplicate eye events
                trial_eye_events.loc[
                    trial_eye_events["eye_event_type"].shift() !=
                    trial_eye_events["eye_event_type"]]

                # Structure eye events dataframe
                trial_eye_events["eye_tracker"] = EYE_TRACKER
                trial_eye_events["experiment_id"] = experiment_id
                trial_eye_events["participant_id"] = experiment_id
                trial_eye_events["filename"] = file
                trial_eye_events["trial_id"] = str(trial_id)
                trial_eye_events["stimuli_module"] = STIMULI_MODULE
                trial_eye_events["stimuli_name"] = stimuli_name
                trial_eye_events["x1"] = np.nan
                trial_eye_events["y1"] = np.nan
                trial_eye_events["token"] = None
                # TODO: Parse pupil data from the dataset
                trial_eye_events["pupil"] = 0
                trial_eye_events["amplitude"] = np.nan
                trial_eye_events["peak_velocity"] = np.nan
                trial_eye_events["peak_velocity"] = np.nan
                trial_eye_events = trial_eye_events[
                    trial_eye_events["eye_event_type"] == "Fixation"]
                trial_eye_events["eye_event_type"] = "fixation"

                eye_events = pd.concat(
                    [eye_events, trial_eye_events])

            # Stop parsing condition
            sample_size -= 1
            if sample_size == 0:
                break

    # Drop unncessary information
    samples.drop(["Computer timestamp [ms]", "Sensor", "Eyetracker timestamp [μs]",
                 "Event", "Event value", "Gaze point X [MCS norm]", "Gaze point Y [MCS norm]",
                  "Gaze point left X [MCS norm]", "Gaze point left Y [MCS norm]",
                  "Gaze point right X [MCS norm]", "Gaze point right Y [MCS norm]",
                  "Fixation point X [MCS norm]", "Fixation point Y [MCS norm]",
                  "Fixation point X [DACS px]", "Fixation point Y [DACS px]"],
                 axis=1, inplace=True)  # TODO: What timestamp are we using right here?

    # Rearrange columns
    eye_events = eye_events[get_eye_event_columns()]
    samples = pd.concat([samples.loc[:, "eye_tracker":],
                        samples.loc[:, : "Eye movement type index"]], axis=1)

    eye_events.reset_index(drop=True, inplace=True)
    samples.reset_index(drop=True, inplace=True)
    return eye_events, samples
