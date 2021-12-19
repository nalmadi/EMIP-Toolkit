import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'EMIP-Toolkit')))
import emip_toolkit as tk
import random as ran
import pytest
import os
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import pandas as pd

emipPath = tk.download("EMIP")

def test_EMIP_dataset():
    EMIP = tk.EMIP_dataset(emipPath+"/EMIP-Toolkit- replication package/emip_dataset/rawdata/", 10)
    assert len(EMIP) == 10
    assert EMIP['177'].trial[0].get_subject_id() == '177'
    assert EMIP['177'].get_number_of_trials() == 7 
    assert EMIP['177'].trial[0].get_sample_number() == 1527


def test_read_SMIRed250():
    filename = emipPath + "/EMIP-Toolkit- replication package/emip_dataset/rawdata/30_rawdata.tsv"
    filetype = 'tsv'
    Ex = tk.read_SMIRed250(filename,filetype)
    assert type(Ex) is tk.Experiment
    assert Ex.get_eye_tracker() == 'SMIRed250'
    assert Ex.filetype == 'tsv'
    assert type(Ex.trial[0]) is tk.Trial
    assert Ex.get_number_of_trials() == 7
    
def test_idt_classifier():
    filename = emipPath + "/EMIP-Toolkit- replication package/emip_dataset/rawdata/30_rawdata.tsv"
    filetype = 'tsv'
    Ex = tk.read_SMIRed250(filename,filetype)
    fixations = Ex.trial[0].get_fixations()
    assert type(fixations[0]) == tk.Fixation
    assert fixations[0].x_cord >= 0.0
    assert fixations[0].x_cord < 1920
    assert fixations[0].y_cord < 1080
    assert fixations[0].y_cord >= 0.0
    
# def test_AlMadi_dataset():
#     ampath = tk.download("AlMadi2018")
#     AM = tk.AlMadi_dataset('../datasets/AlMadi2018/AlMadi2018/ASCII', 10)
#     assert len(AM) == 10
#     assert AM['99'].trial[0].get_subject_id() == '99'
#     assert AM['99'].get_number_of_trials() == 7 
#     assert AM['99'].trial[0].get_sample_number() == 1399

def test_find_background_color():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = '../datasets/AlMadi2018/ASCII/runtime/images/5659673816338139458.png'
    abs_file_path = os.path.join(script_dir, rel_path)
    file = Image.open(abs_file_path)
    color = tk.find_background_color(file)
    assert type(color) == str
    assert color == "black" or color == "white"

def test_find_aoi():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = '../datasets/AlMadi2018/ASCII/runtime/images/5659673816338139458.png'
    abs_file_path = os.path.join(script_dir, rel_path)
    file = Image.open(abs_file_path)
    aoi = tk.find_aoi(img = file)
    assert type(aoi) == pd.core.frame.DataFrame
    assert all([a == b for a,b in zip(aoi.keys().values,['kind', 'name', 'x', 'y', 'width', 'height', 'image'])])
    # More tests on column values

def test_draw_aoi():
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = '../datasets/AlMadi2018/ASCII/runtime/images/5659673816338139458.png'
    abs_file_path = os.path.join(script_dir, rel_path)
    image = Image.open(abs_file_path)
    aoi = tk.find_aoi(img = image)
    rect_img = tk.draw_aoi(aoi, abs_file_path, "")
    assert type(rect_img) == Image.Image

def test_add_tokens_to_AOI():
    file_path = '../emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/'
    aois_raw = tk.find_aoi(image= "vehicle_java.jpg" , image_path = './emip_dataset/stimuli/' )
    aois_tokens = tk.add_tokens_to_AOIs(file_path,aois_raw)
    assert type(aois_tokens) == pd.core.frame.DataFrame
    assert aois_tokens["image"][1] == "vehicle_java.jpg"
    assert len(aois_tokens["token"]) == 112

def test_add_srcml_to_AOIs():
    file_path = '../emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/'
    aois_raw = tk.find_aoi(image= "vehicle_java.jpg" , image_path = './emip_dataset/stimuli/' )
    srcML_path = '../datasets/EMIP2021/'
    AOIs_wSRCs= tk.add_srcml_to_AOIs(aois_raw, srcML_path)
    
    assert type(AOIs_wSRCs) == pd.core.frame.DataFrame
    assert type(AOIs_wSRCs["srcML_tag"]) == pd.Series
    assert len(AOIs_wSRCs["srcML_tag"]) != 0
    assert type(AOIs_wSRCs["srcML_tag"][1]) == str
