import emip_toolkit as tk


def test_reading_raw_data_files():
    EMIP = tk.EMIP_dataset('../../emip_dataset/rawdata/', 10)
    assert len(EMIP) == 10
    assert EMIP['99'].trial[0].get_subject_id() == 99
    assert EMIP['99'].get_number_of_trials() == 7 
    assert EMIP['99'].trial[0].get_sample_number() == 1399

# def test_AlMadi_dataset():

# def test_fixation_class():

# def test_saccade_class():

# def test_blink_class():

# def test_trial_class():

# def test_experiment_class():

# def test_idt_classifer():

# def test_read_SMIRed250():

# def test_read_EyeLink1000():

# def test_find_background_color():

# def test_find_aoi():

# def test_add_tokens_to_AOIs():

# def test_add_srcml_to_AOIs():

# def test_overlap():

# def test_hit_test():

