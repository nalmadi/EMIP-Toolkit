from emtk.parsers import AlMadi

def test_AlMadi():
    eye_events, _ = AlMadi(3)

    assert len(eye_events.columns) == 18
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

    assert len( eye_events["experiment_id"].unique() ) == 3

    experiment_id = '001'
    trial_id = '1'

    experiment_001 = eye_events.loc[eye_events["experiment_id"] == experiment_id]
    assert len( experiment_001["trial_id"].unique() ) == 16


    fixations = eye_events.loc[(eye_events["experiment_id"] == experiment_id) & \
                            (eye_events["trial_id"] == trial_id) & \
                            (eye_events["eye_event_type"] == "fixation")]

    assert fixations.shape[0] == 14
    assert len( fixations["stimuli_name"].unique() ) == 1
    assert fixations["stimuli_name"].unique()[0] == "5667346413132987794.png"

    # TODO: Check dtype of each column  