import pandas as pd

def _get_meta_data(eye_events: pd.DataFrame, 
                   eye_tracker_col: str = "eye_tracker",
                   stimuli_module_col: str = "stimuli_module", 
                   stimuli_name_col: str = "stimuli_name") -> tuple:

    col_names = [ eye_tracker_col, stimuli_module_col, stimuli_name_col ]

    for col in col_names:
        if len( eye_events[col].unique() ) > 1 :
            raise Exception("Error, there are more than " +
                            "one unique value in {col} column".format(col = col))


    return ( eye_events[col].unique()[0] for col in col_names )

    