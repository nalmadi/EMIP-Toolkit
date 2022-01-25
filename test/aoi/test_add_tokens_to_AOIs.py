from emtk.aoi import find_aoi, add_tokens_to_AOIs
from emtk.parsers import EMIP

def test_add_tokens_to_AOIs():

    eye_events, _ = EMIP(8)

    trial_eye_events = eye_events.loc[ ( eye_events["experiment_id"] == '106' ) & \
                                        ( eye_events["trial_id"] == "2" ) ]

    aoi = find_aoi(trial_eye_events)
    code_file_path = "emtk/datasets/emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = add_tokens_to_AOIs(code_file_path, aoi)

    assert aois_with_tokens["token"][31] == "this.type"
    assert aois_with_tokens["token"][85] == "("