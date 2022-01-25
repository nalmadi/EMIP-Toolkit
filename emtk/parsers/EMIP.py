import os
from random import sample
import pandas as pd

from .eye_events import eye_event_list, get_eye_event_columns
from .samples import get_samples_columns, samples_list
from .download import download

from emtk.fixation_classification import idt_classifier

EYE_TRACKER = "SMIRed250"
FILE_TYPE = ".tsv"
# Uncomment the next 2 lines with the 2-line comment below 
# when issue #70 is resolved: 
# https://github.com/nalmadi/EMIP-Toolkit/issues/70

RAWDATA_MODULE = "emtk/datasets/emip_dataset/rawdata"
STIMULI_MODULE = "emtk/datasets/emip_dataset/stimuli"

# RAWDATA_MODULE = "emtk/datasets/EMIP/rawdata"
# STIMULI_MODULE = "emtk/datasets/EMIP/stimuli"

SAMPLE_BASE_COLUMNS = ['Time', 'Type', 'Trial', 'L Raw X [px]', 'L Raw Y [px]', 'R Raw X [px]', 
                    'R Raw Y [px]', 'L Dia X [px]', 'L Dia Y [px]', 'L Mapped Diameter [mm]', 
                    'R Dia X [px]', 'R Dia Y [px]', 'R Mapped Diameter [mm]', 'L CR1 X [px]', 
                    'L CR1 Y [px]', 'L CR2 X [px]', 'L CR2 Y [px]', 'R CR1 X [px]', 'R CR1 Y [px]', 
                    'R CR2 X [px]', 'R CR2 Y [px]', 'L POR X [px]', 'L POR Y [px]', 'R POR X [px]', 
                    'R POR Y [px]', 'Timing', 'L Validity', 'R Validity', 'Pupil Confidence', 
                    'L Plane', 'R Plane', 'L EPOS X', 'L EPOS Y', 'L EPOS Z', 'R EPOS X', 'R EPOS Y', 
                    'R EPOS Z', 'L GVEC X', 'L GVEC Y', 'L GVEC Z', 'R GVEC X', 'R GVEC Y', 
                    'R GVEC Z', 'Frame', 'Aux1']

def EMIP(sample_size: int = 216):
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
    samples = []
    parsed_experiments = []

    # Uncomment the next 2 lines when issue #70 is resolved: 
    # https://github.com/nalmadi/EMIP-Toolkit/issues/70
    # if not os.path.isfile(RAWDATA_MODULE):
    #     download("EMIP")
    
    # go over .tsv files in the rawdata directory add files and count them
    # r = root, d = directories, f = files
    for r, _, f in os.walk(RAWDATA_MODULE):
        for file in f:
            if '.tsv' in file:
                experiment_id = file.split('/')[-1].split('_')[0]

                if  experiment_id not in parsed_experiments:

                    parsed_experiments.append(experiment_id)

                    new_eye_events, new_samples = read_SMIRed250(
                        root_dir = r,
                        filename =  file,
                        experiment_id = experiment_id, 
                    )

                    eye_events.extend(new_eye_events)
                    samples.extend(new_samples)

                else:
                    print("Error, experiment already in dictionary")

            sample_size -= 1
            if sample_size == 0:
                break

    eye_events_df = pd.DataFrame(eye_events, columns = get_eye_event_columns())

    # Convert columns with numbers formatted as strings to dtype of numeric
    samples_df = pd.DataFrame(samples, columns = get_samples_columns(SAMPLE_BASE_COLUMNS))
    id_dfs = samples_df[["experiment_id", "participant_id", "trial_id"]]
    samples_df = samples_df.apply(pd.to_numeric, errors='ignore')
    samples_df[id_dfs.columns] = id_dfs

    return eye_events_df, samples_df


def read_SMIRed250(root_dir, filename, experiment_id,
                   minimum_duration=50, sample_duration=4, maximum_dispersion=25):
    """Read tsv file from SMI Red 250 eye tracker

    Parameters
    ----------
    filename : str
        name of the tsv file

    minimum_duration : int, optional
        minimum duration for a fixation in milliseconds, less than minimum is considered noise.
        set to 50 milliseconds by default.

    sample_duration : int, optional
        Sample duration in milliseconds, this is 4 milliseconds based on this eye tracker.

    maximum_dispersion : int, optional
        maximum distance from a group of samples to be considered a single fixation.
        Set to 25 pixels by default.

    Returns
    -------
    Experiment
        an Experiment object from SMIRed250 data
    """

    # Reads raw data and sets up
    tsv_file = open(os.path.join(root_dir, filename))
    print("parsing file:", filename.split("/")[-1])
    text = tsv_file.read()
    text_lines = text.split('\n')

    trial_id = 0
    stimuli_name = ""
    raw_fixations = []
    active = False  # Indicates whether samples are being recorded in trials
                    # The goal is to skip metadata in the file

    eye_events = []
    samples = []

    # Parses the data into dataframes
    for line in text_lines:
        token = line.split("\t")

        if len(token) < 3:
            continue

        if active:
            # Filter MSG samples if any exist, or R eye is inValid
            if token[1] == "SMP" and token[27] != "-1":
                # Get x and y for each sample (right eye only)
                # [23] R POR X [px]	 '0.00',
                # [24] R POR Y [px]	 '0.00',

                new_sample = samples_list(
                    eye_tracker=EYE_TRACKER, 
                    experiment_id=experiment_id,
                    participant_id=experiment_id, 
                    filename=filename,
                    trial_id=str(trial_id), 
                    stimuli_module=STIMULI_MODULE,
                    stimuli_name=stimuli_name, 
                    token=token
                )

                samples.append(new_sample)

                raw_fixations.append([int(token[0]), float(token[23]), float(token[24])])

        if token[1] == "MSG" and token[3].find(".jpg") != -1:
            
            if active:
                filter_eye_events = idt_classifier(raw_fixations=raw_fixations,
                                                  minimum_duration=minimum_duration,
                                                  sample_duration=sample_duration,
                                                  maximum_dispersion=maximum_dispersion)
                # TODO saccades
                
                for timestamp, duration, x_cord, y_cord in filter_eye_events:
            
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
                                                    token=token,
                                                    pupil=0,
                                                    eye_event_type="fixation")

                    eye_events.append(new_eye_event)
               
                trial_id += 1
            
            stimuli_name = token[3].split(' ')[-1]  # Message: vehicle_java2.jpg
            raw_fixations = []
            active = True

    # Adds the last trial
    filter_fixations = idt_classifier(raw_fixations=raw_fixations,
                                      minimum_duration=minimum_duration,
                                      sample_duration=sample_duration,
                                      maximum_dispersion=maximum_dispersion)

    for timestamp, duration, x_cord, y_cord in filter_fixations:
        
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
                                                    token=token,
                                                    pupil=0,
                                                    eye_event_type="fixation")

        eye_events.append(new_eye_event)                                      
    
    return eye_events, samples