import os
import pandas as pd
import numpy as np

from .eye_events import get_eye_event_columns, eye_event_list
from .download import download

EYE_TRACKER = "EyeLink1000"
FILE_eye_event_type = ".tsv"
RAWDATA_MODULE = "emtk/datasets/AlMadi2018/ASCII/"
STIMULI_MODULE = "emtk/datasets/AlMadi2018/runtime/images/"

def AlMadi(sample_size=216):
    """Import the EMIP dataset

    Parameters
    ----------
    sample_size : int, optional
        the number of subjects to be processed, the default is 216.

    Returns
    -------
    pandas.DataFrame
        dataFrame of eye events from every experiment in the dataset.
    """
    eye_events = []
    parsed_experiments = []

    if not os.path.isfile(RAWDATA_MODULE):
        download("AlMadi2018")

    # go over .tsv files in the rawdata directory add files and count them
    # r = root, d = directories, f = files
    for r, _, f in os.walk(RAWDATA_MODULE):
        for file in f:
            if '.asc' in file:

                experiment_id = file.split('.')[0].split('/')[-1]

                if  experiment_id not in parsed_experiments:

                    parsed_experiments.append(experiment_id)

                    new_eye_events = read_EyeLink1000(
                        root_dir = r,
                        filename =  file,
                        experiment_id = experiment_id
                    )
                
                    eye_events.extend(new_eye_events)

                else:
                    print("Error, experiment already in dictionary")


            sample_size -= 1
            # breaks after sample_size
            if sample_size == 0:
                break

    eye_events_df = pd.DataFrame(eye_events, columns = get_eye_event_columns())
    
    return eye_events_df, pd.DataFrame()



def read_EyeLink1000(root_dir, filename, experiment_id):
    """Read asc file from Eye Link 1000 eye tracker

    Parameters
    ----------
    filename : str
        name of the asc file
        
    fileeye_event_type : str
        fileeye_event_type of the file, e.g. "tsv"
        
    Returns
    -------
    Experiment
        an Experiment object of EyeLink1000 data
    """

    asc_file = open(os.path.join(root_dir, filename))
    print("parsing file:", filename)

    trial_id = 0
    eye_events = []

    text = asc_file.read()
    text_lines = text.split('\n')

    for line in text_lines:

        token = line.split()

        if not token:
            continue

        if "TRIALID" in token:
            # List of eye events
            trial_id = int(token[-1])

            # Read stimuli location
            index = str(int(trial_id) + 1)
            location = 'emtk/datasets/AlMadi2018/runtime/dataviewer/' + experiment_id + \
                        '/graphics/VC_' + index + '.vcl'
                      
            with open(location, 'r') as file:
                stimuli_name = file.readlines()[1].split()[-3].split('/')[-1]

            trial_id = int(token[-1])

        if token[0] not in ("EFIX", "ESACC", "EBLINK"):
            continue

        x_cord = y_cord = x1_cord = y1_cord = pupil = amplitude = peak_velocity = np.nan
        eye_event_type = "blink"

        if token[0] == "EFIX":
            timestamp = int(token[2])
            duration = int(token[4])
            x_cord = float(token[5])
            y_cord = float(token[6])
            pupil = int(token[7])
            eye_event_type = "fixation"

        if token[0] == "ESACC":
            timestamp = int(token[2])
            duration = int(token[4])
            x_cord = float(token[5]) if token[5] != '.' else 0.0
            y_cord = float(token[6]) if token[6] != '.' else 0.0
            x1_cord = float(token[7]) if token[7] != '.' else 0.0
            y1_cord = float(token[8]) if token[8] != '.' else 0.0
            amplitude = float(token[9])
            peak_velocity = int(token[10])
            eye_event_type = "saccade"

        if token[0] == "EBLINK":
            timestamp = int(token[2])
            duration = int(token[4])
            eye_event_type = "blink"

        new_eye_event = eye_event_list(eye_tracker=EYE_TRACKER, 
                                    experiment_id=experiment_id,
                                    participant_id=experiment_id, 
                                    filename=filename,
                                    trial_id=str(trial_id), 
                                    stimuli_module=STIMULI_MODULE,
                                    stimuli_name=stimuli_name, 
                                    duration=duration, 
                                    timestamp=timestamp,
                                    x0=x_cord, 
                                    y0=y_cord,
                                    x1=x1_cord, 
                                    y1=y1_cord, 
                                    token=token, 
                                    pupil=pupil,
                                    amplitude=amplitude, 
                                    peak_velocity=peak_velocity, 
                                    eye_event_type=eye_event_type)

        eye_events.append(new_eye_event)

    asc_file.close()

    return eye_events