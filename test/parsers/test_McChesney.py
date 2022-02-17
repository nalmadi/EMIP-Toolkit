from emtk.parsers import McChesney

def test_McChesney():
    eye_events, samples = McChesney(5)

    assert len(eye_events.columns) == 18

    assert len( eye_events["experiment_id"].unique() ) == 5

    experiment_id = "P131"
    trial_id = "2"

    experiment_P131 = eye_events.loc[eye_events["experiment_id"] == experiment_id]
    assert len( experiment_P131["trial_id"].unique() ) == 3


    fixations = eye_events.loc[(eye_events["experiment_id"] == experiment_id) & \
                            (eye_events["trial_id"] == trial_id) & \
                            (eye_events["eye_event_type"] == "fixation")]

    assert fixations.shape[0] == 3977
    assert len( fixations["stimuli_name"].unique() ) == 1
    assert fixations["stimuli_name"].unique()[0] == "02_P2Sa.png"

    trial_samples = samples.loc[(samples["experiment_id"] == experiment_id) & \
                    (samples["trial_id"] == trial_id)]
    assert trial_samples.shape[0] == 8799

    # TODO: Check dtype of each column  