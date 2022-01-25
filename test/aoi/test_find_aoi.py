from emtk.aoi import find_aoi
from emtk.parsers import EMIP
from emtk.util import _get_stimuli, _get_meta_data

def test_find_aoi():

    eye_events, _ = EMIP(8)

    trial_eye_events = eye_events.loc[ ( eye_events["experiment_id"] == '106' ) & \
                                        ( eye_events["trial_id"] == "2" ) ]

    aoi = find_aoi(trial_eye_events)
    assert aoi["kind"].unique() == "sub-line"
    assert aoi["name"][78] == "line 16 part 2"
    assert aoi["x"][3] == 853.5	
    assert aoi["y"][12] == 214	
    assert aoi["width"][39] == 168
    assert aoi["height"][57] == 22
    assert aoi["image"].unique() == "vehicle_java2.jpg"

    eye_tracker, stimuli_module, stimuli_name = _get_meta_data(trial_eye_events)
    image = _get_stimuli(stimuli_module, stimuli_name, eye_tracker)
    aoi = find_aoi(image = image)
    assert aoi["kind"].unique() == "sub-line"
    assert aoi["name"][78] == "line 16 part 2"
    assert aoi["x"][3] == 853.5	
    assert aoi["y"][12] == 214	
    assert aoi["width"][39] == 168
    assert aoi["height"][57] == 22
    assert aoi["image"].unique() == ""

    #TODO: Test with Al Madi dataset