from emtk.aoi import find_aoi, add_tokens_to_AOIs, add_srcml_to_AOIs
from emtk.parsers import EMIP

def test_add_tokens_to_AOIs():

    eye_events, _ = EMIP(8)

    trial_eye_events = eye_events.loc[ ( eye_events["experiment_id"] == "106" ) & \
                                        ( eye_events["trial_id"] == "2" ) ]

    aoi = find_aoi(trial_eye_events)
    code_file_path = "emtk/datasets/emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = add_tokens_to_AOIs(code_file_path, aoi)

    srcML_path = "emtk/datasets/EMIP2021/"
    aois_tokens_srcml = add_srcml_to_AOIs(aois_with_tokens, srcML_path)

    assert aois_tokens_srcml["srcML_tag"][4] == "class->block->decl_stmt->decl->type->name"
    assert aois_tokens_srcml["srcML_tag"][59] == \
        "class->block->function->block->block_content->if_stmt->if->condition->expr->operator"