'''
John, Henry, Ryan
Pytest for EMIP Toolkit
CS321 Final Project
December 7, 2021
'''
import emip_toolkit as tk

def test_EMIP_dataset():
    '''Testing reading raw files from EMIP Dataset'''
    
    EMIP = tk.EMIP_dataset('./emip_dataset/rawdata/', 10)

    assert len(EMIP)==10
    assert EMIP['100'].trial[0].get_subject_id()=='100'
    assert EMIP['100'].get_number_of_trials()==7
    assert EMIP['100'].trial[0].get_sample_number()==12040


def test_fixation_filter():
    '''Tests getting a specific trial and subject and number of fixations'''
    
    EMIP = tk.EMIP_dataset('./emip_dataset/rawdata/', 10)
    subject_ID = '106'
    trial_num = 2 

    assert EMIP[subject_ID].trial[trial_num].get_fixation_number() == 357   
    assert EMIP[subject_ID].trial[trial_num].get_sample_number() == 18964
    assert EMIP[subject_ID].trial[trial_num].get_trial_image() == "vehicle_java2.jpg"
    
    
def test_offset():
    '''Testing the offset functionality'''

    EMIP = tk.EMIP_dataset('./emip_dataset/rawdata/', 10)
    subject_ID ='106'
    trial_num = 2
    image_path = "../../emip_dataset/stimuli/"
    EMIP[subject_ID].trial[trial_num].sample_offset(-200, 100)
    assert (EMIP[subject_ID].trial[trial_num].get_offset()==(-200,100))
    
    EMIP[subject_ID].trial[trial_num].reset_offset()
    assert EMIP[subject_ID].trial[trial_num].get_offset()==(0,0)

def test_aoi():
    '''Testing aoi creation'''
    image_path = "./emip_dataset/stimuli/"
    image = "rectangle_java2.jpg"
    aoi = tk.find_aoi(image, image_path, level="sub-line")
    
    assert(aoi['x'][0]==589.5)
    assert(aoi['image'][0]=="rectangle_java2.jpg")
    
def test_add_token():
    '''Testing token generated aois'''
    
    image_path = "./emip_dataset/stimuli/"
    image = "rectangle_java2.jpg"
    aoi = tk.find_aoi(image, image_path, level="sub-line")
    file_path = "./emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)
    
    assert aois_with_tokens['token'][0]=='public'


def test_add_tags_and_tokens():
    '''Tests the adding of the tags to AOIs and tokens'''

    image_path = "./emip_dataset/stimuli/"
    image = "rectangle_java2.jpg"
    aoi = tk.find_aoi(image, image_path, level="sub-line")
    file_path = "./emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)

    srcML_path = "./datasets/EMIP2021/"
    aois_tokens_srcml = tk.add_srcml_to_AOIs(aois_with_tokens, srcML_path)

    assert aois_tokens_srcml['srcML_tag'][0] == 'class->specifier'

    

