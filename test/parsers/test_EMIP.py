from emtk.parsers import EMIP

def test_EMIP():
    eye_events, samples = EMIP(8)

    assert len(eye_events.columns) == 18
    assert len(samples.columns) == 52
    assert list(eye_events.columns) == [
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
    assert list(samples.columns) == [
        "eye_tracker",                            
        "experiment_id",
        "participant_id",
        "filename",
        "trial_id",
        "stimuli_module",
        "stimuli_name",
        'Time', 'Type', 'Trial', 'L Raw X [px]', 'L Raw Y [px]', 'R Raw X [px]', 
        'R Raw Y [px]', 'L Dia X [px]', 'L Dia Y [px]', 'L Mapped Diameter [mm]', 
        'R Dia X [px]', 'R Dia Y [px]', 'R Mapped Diameter [mm]', 'L CR1 X [px]', 
        'L CR1 Y [px]', 'L CR2 X [px]', 'L CR2 Y [px]', 'R CR1 X [px]', 'R CR1 Y [px]', 
        'R CR2 X [px]', 'R CR2 Y [px]', 'L POR X [px]', 'L POR Y [px]', 'R POR X [px]', 
        'R POR Y [px]', 'Timing', 'L Validity', 'R Validity', 'Pupil Confidence', 
        'L Plane', 'R Plane', 'L EPOS X', 'L EPOS Y', 'L EPOS Z', 'R EPOS X', 'R EPOS Y', 
        'R EPOS Z', 'L GVEC X', 'L GVEC Y', 'L GVEC Z', 'R GVEC X', 'R GVEC Y', 
        'R GVEC Z', 'Frame', 'Aux1'
    ]

    assert len( eye_events["experiment_id"].unique() ) == 8

    experiment_id = '100'
    experiment_100 = eye_events.loc[eye_events["experiment_id"] == experiment_id]
    assert len( experiment_100["trial_id"].unique() ) == 7


    experiment_id = '106'
    trial_id = '2'

    fixations = eye_events.loc[(eye_events["experiment_id"] == experiment_id) & \
                            (eye_events["trial_id"] == trial_id) & \
                            (eye_events["eye_event_type"] == "fixation")]
    assert fixations.shape[0] == 357
    assert len( fixations["stimuli_name"].unique() ) == 1
    assert fixations["stimuli_name"].unique()[0] == "vehicle_java2.jpg"

    trial_samples = samples.loc[(samples["experiment_id"] == experiment_id) & \
                        (samples["trial_id"] == trial_id)]
    assert trial_samples.shape[0] == 18964

    # TODO: Check dtype of each column                        
