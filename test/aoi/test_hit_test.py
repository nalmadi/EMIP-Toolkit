from emtk.aoi import find_aoi, add_tokens_to_AOIs, add_srcml_to_AOIs, hit_test
from emtk.parsers import EMIP

def test_add_tokens_to_AOIs():

    eye_events, _ = EMIP(8)

    trial_eye_events = eye_events.loc[ ( eye_events["experiment_id"] == '106' ) & \
                                        ( eye_events["trial_id"] == "2" ) ]

    aoi = find_aoi(trial_eye_events)
    code_file_path = "emtk/datasets/emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = add_tokens_to_AOIs(code_file_path, aoi)

    srcML_path = "emtk/datasets/EMIP2021/"
    aois_tokens_srcml = add_srcml_to_AOIs(aois_with_tokens, srcML_path)

    assert aois_tokens_srcml['srcML_tag'][0] == 'class->specifier'

    trial_fixations = trial_eye_events.loc[trial_eye_events["eye_event_type"] == "fixation"]

    aoi_fixes = hit_test(trial_fixations, aois_tokens_srcml)

    assert aoi_fixes["experiment_id"].unique() == "106"
    assert aoi_fixes["trial_id"].unique() == "2"
    assert aoi_fixes["stimuli_name"].unique() == "vehicle_java2.jpg"
    assert aoi_fixes["timestamp"][31] == 21899618891
    assert aoi_fixes["duration"][82] == 96
    assert aoi_fixes["x0"][89] == 784.205
    assert aoi_fixes["y0"][103] == 501.253125
    assert aoi_fixes["aoi_kind"].unique() == "sub-line"
    assert aoi_fixes["aoi_name"][105] == "line 10 part 3"
    assert aoi_fixes['aoi_y'][67] == 303
    assert aoi_fixes['aoi_width'][68] == 129
    assert aoi_fixes['aoi_height'][72] == 22
    assert aoi_fixes['aoi_token'][134] == 'this.currentSpeed'
    assert aoi_fixes['aoi_srcML_tag'][157] == \
        "class->block->function->block->block_content->if_stmt->if->condition->expr->name->name->operator->name"