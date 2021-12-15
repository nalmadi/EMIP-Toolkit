import emip_toolkit as tk


def test_reading_raw_data_files():
    EMIP = tk.EMIP_dataset('../../emip_dataset/rawdata/', 10)
    assert len(EMIP) == 10
    assert EMIP['99'].trial[0].get_subject_id() == 99
    assert EMIP['99'].get_number_of_trials() == 7 
    assert EMIP['99'].trial[0].get_sample_number() == 1399


def test_fixation_accessor():
    
