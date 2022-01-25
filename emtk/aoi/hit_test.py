import pandas as pd

def hit_test(fixations: pd.DataFrame, aoi_df: pd.DataFrame, radius: int = 25,
             eye_tracker_col = "eye_tracker", experiment_id_col = "experiment_id", 
             participant_id_col = "participant_id", filename_col = "filename",
             trial_id_col = "trial_id", stimuli_module_col = "stimuli_module", 
             stimuli_name_col = "stimuli_name", timestamp_col = "timestamp",
             duration_col = "duration",
             fixation_x0_col: str = "x0", fixation_y0_col: str = "y0",
             aoi_kind_col: str = "kind", aoi_name_col: str = "name",
             aoi_x_col: str = "x", aoi_y_col: str = "y", 
             aoi_width_col: str = "width", aoi_height_col: str = "height",
             aoi_token_col: str = "token", aoi_srcML_tag_col: str = "srcML_tag") -> pd.DataFrame:

    _fixations = fixations.copy()[[
        eye_tracker_col,                            
        experiment_id_col,
        participant_id_col,
        filename_col,
        trial_id_col,
        stimuli_module_col,
        stimuli_name_col,
        timestamp_col,
        duration_col,
        fixation_x0_col,
        fixation_y0_col,
    ]]

    _aoi_df = aoi_df.copy()[[aoi_kind_col, aoi_name_col, aoi_x_col, aoi_y_col, 
                    aoi_width_col, aoi_height_col, aoi_token_col, aoi_srcML_tag_col]]

    _fixations['_name'] = \
        _fixations.apply(lambda _fixation_row: _hit_test(_fixation_row, 
                                                         _aoi_df, 
                                                         radius,
                                                         fixation_x0_col,
                                                         fixation_y0_col,
                                                         aoi_x_col,
                                                         aoi_y_col,
                                                         aoi_width_col,
                                                         aoi_height_col,
                                                         aoi_name_col),
                         axis = 1)

    return _fixations.merge(_aoi_df.add_prefix("aoi_"), left_on = "_name", 
                            right_on = "aoi_name", how="inner").drop("_name", axis = 1)                                                    



def _hit_test(_fixation_row: pd.DataFrame, aoi_df: pd.DataFrame, radius: int = 25,
              fixation_x0_col: str = "x0", fixation_y0_col: str = "y0",
              aoi_x_col: str = "x", aoi_y_col: str = "y", 
              aoi_width_col: str = "width", aoi_height_col: str = "height",
              aoi_name_col: str = "name") -> pd.DataFrame:
    
    for _, aoi_row in aoi_df.iterrows():
        box_x = aoi_row[aoi_x_col] - (radius / 2)
        box_y = aoi_row[aoi_y_col] - (radius / 2)
        box_w = aoi_row[aoi_width_col] + (radius / 2)
        box_h = aoi_row[aoi_height_col] + (radius / 2)

        if box_x <= _fixation_row[fixation_x0_col] <= box_x + box_w and \
            box_y <= _fixation_row[fixation_y0_col] <= box_y + box_h:
            return aoi_row[aoi_name_col]