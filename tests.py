import emip_toolkit as tk
import random as ran
import pytest
import os
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import pandas as pd

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
    assert fixations[0].x_cord < 1920
    assert fixations[0].y_cord < 1080
    assert fixations[0].y_cord >= 0.0
    
    #for i in range(0,200):

# def test_AlMadi_dataset():
#     AM = tk.AlMadi_dataset('datasets/AlMadi2018/ASCII/001.asc', 10)
#     assert len(AM) == 10
#     assert AM['99'].trial[0].get_subject_id() == '99'
#     assert AM['99'].get_number_of_trials() == 7 
#     assert AM['99'].trial[0].get_sample_number() == 1399

def test_find_background_color():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = './datasets/AlMadi2018/ASCII/runtime/images/5659673816338139458.png'
    abs_file_path = os.path.join(script_dir, rel_path)
    file = Image.open(abs_file_path)
    color = tk.find_background_color(file)
    assert type(color) == str
    assert color == "black" or color == "white"

def test_find_aoi():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = './datasets/AlMadi2018/ASCII/runtime/images/5659673816338139458.png'
    abs_file_path = os.path.join(script_dir, rel_path)
    file = Image.open(abs_file_path)
    aoi = tk.find_aoi(img = file)
    assert type(aoi) == pd.core.frame.DataFrame
    assert all([a == b for a,b in zip(aoi.keys().values,['kind', 'name', 'x', 'y', 'width', 'height', 'image'])])
    # More tests on column values

def test_draw_aoi():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = './datasets/AlMadi2018/ASCII/runtime/images/5659673816338139458.png'
    abs_file_path = os.path.join(script_dir, rel_path)
    image = Image.open(abs_file_path)
    aoi = tk.find_aoi(img = image)
    rect_img = tk.draw_aoi(aoi, abs_file_path, "")
    assert type(rect_img) == Image.Image


# def test_fixation_class():

# def test_saccade_class():

# def test_blink_class():

# def test_trial_class():

# def test_experiment_class():

# def test_read_EyeLink1000():

# def test_add_tokens_to_AOIs():

# def test_add_srcml_to_AOIs():

# def test_overlap():

# def test_hit_test():

