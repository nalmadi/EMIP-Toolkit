import emip_toolkit as tk
import random as ran
import pytest

@pytest.mark.skip(reason="not needed at this time")
def test_EMIP_dataset():
    EMIP = tk.EMIP_dataset('../emip_dataset/rawdata/', 10)
    assert len(EMIP) == 10
    assert EMIP['99'].trial[0].get_subject_id() == '99'
    assert EMIP['99'].get_number_of_trials() == 7 
    assert EMIP['99'].trial[0].get_sample_number() == 1399


def test_read_SMIRed250():
    filename = '../emip_dataset/rawdata/1_rawdata.tsv'
    filetype = 'tsv'
    Ex = tk.read_SMIRed250(filename,filetype)
    assert type(Ex) is tk.Experiment
    assert Ex.get_eye_tracker() == 'SMIRed250'
    assert Ex.filetype == 'tsv'
    assert type(Ex.trial[0]) is tk.Trial
    assert Ex.get_number_of_trials() == 7
    

def test_idt_classifier():
    filename = '../emip_dataset/rawdata/1_rawdata.tsv'
    filetype = 'tsv'
    Ex = tk.read_SMIRed250(filename,filetype)
    fixations = Ex.trial[0].get_fixations()
    assert type(fixations[0]) == tk.Fixation
    assert fixations[0].x_cord >= 0.0
    assert fixations[0].y_cord >= 0.0
    assert fixations

    
    #for i in range(0,200):


    #raw_fixations =[[0,1.2,1.3],[1,-5,-4],[2,1925,1089]]





     


# def test_fixation_class():

# def test_saccade_class():

# def test_blink_class():

# def test_trial_class():

# def test_experiment_class():

# def test_idt_classifer():

#def test_read_SMIRed250():

# def test_read_EyeLink1000():

# def test_find_background_color():

# def test_find_aoi():

# def test_add_tokens_to_AOIs():

# def test_add_srcml_to_AOIs():

# def test_overlap():

# def test_hit_test():

