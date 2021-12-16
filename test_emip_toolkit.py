'''
John, Henry, Ryan
Pytest for EMIP Toolkit
CS321 Final Project
December 7, 2021
'''
import emip_toolkit as tk

emipPath = tk.download("EMIP")
EMIP = tk.EMIP_dataset(emipPath+"/EMIP-Toolkit- replication package/emip_dataset/rawdata/", 5)
# def test_download

def test_EMIP_dataset():
    '''Testing reading raw files from EMIP Dataset'''
   
    assert len(EMIP)==5
    assert EMIP['177'].trial[0].get_subject_id()=='177'
    assert EMIP['177'].get_number_of_trials()==7
    assert EMIP['177'].trial[0].get_sample_number()==1527


def test_fixation_filter():
    '''Tests getting a specific trial and subject and number of fixations'''
    
    #EMIP = tk.EMIP_dataset('./emip_dataset/testdata/', 10)
    subject_ID = '6'
    trial_num = 2 

    assert EMIP[subject_ID].trial[trial_num].get_fixation_number() == 241   
    assert EMIP[subject_ID].trial[trial_num].get_sample_number() == 9952
    assert EMIP[subject_ID].trial[trial_num].get_trial_image() == "rectangle_java.jpg"
    
    
def test_offset():
    '''Testing the offset functionality'''

    #EMIP = tk.EMIP_dataset('./emip_dataset/testdata/', 10)
    subject_ID ='6'
    trial_num = 2
    image_path = "../../emip_dataset/stimuli/"
    EMIP[subject_ID].trial[trial_num].sample_offset(-200, 100)
    assert (EMIP[subject_ID].trial[trial_num].get_offset()==(-200,100))
    
    EMIP[subject_ID].trial[trial_num].reset_offset()
    assert EMIP[subject_ID].trial[trial_num].get_offset()==(0,0)

def test_aoi():
    '''Testing aoi creation'''
    image_path = emipPath+"/EMIP-Toolkit- replication package/emip_dataset/stimuli/"
    image = "rectangle_java2.jpg"
    aoi = tk.find_aoi(image, image_path, level="sub-line")
    
    assert(aoi['x'][0]==589.5)
    assert(aoi['image'][0]=="rectangle_java2.jpg")
    
def test_add_token():
    '''Testing token generated aois'''
    
    image_path = emipPath+"/EMIP-Toolkit- replication package/emip_dataset/stimuli/"
    image = "rectangle_java2.jpg"
    aoi = tk.find_aoi(image, image_path, level="sub-line")
    file_path = emipPath+"/EMIP-Toolkit- replication package/emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)
    
    assert aois_with_tokens['token'][0]=='public'


def test_add_tags_and_tokens():
    '''Tests the adding of the tags to AOIs and tokens'''

    image_path = emipPath+"/EMIP-Toolkit- replication package/emip_dataset/stimuli/"
    image = "rectangle_java.jpg"
    aoi = tk.find_aoi(image, image_path, level="sub-line")
    file_path = emipPath+"/EMIP-Toolkit- replication package/emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)

    srcML_path = "./datasets/EMIP2021/"
    aois_tokens_srcml = tk.add_srcml_to_AOIs(aois_with_tokens, srcML_path)

    assert aois_tokens_srcml['srcML_tag'][0] == 'class->specifier'


def test_hit_test():
    '''Tests the hit test between fixation and aois'''

    #EMIP = tk.EMIP_dataset('./emip_dataset/testdata/', 10)

    subject_ID = '6'
    trial_num = 2 

    image_path = emipPath+"/EMIP-Toolkit- replication package/emip_dataset/stimuli/"
    image = "rectangle_java2.jpg"
    aoi = tk.find_aoi(image, image_path, level="sub-line")
    file_path = emipPath+"/EMIP-Toolkit- replication package/emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)
    srcML_path = "./datasets/EMIP2021/"
    
    aois_tokens_srcml = tk.add_srcml_to_AOIs(aois_with_tokens, srcML_path)
    aoi_fixes = tk.hit_test(EMIP[subject_ID].trial[trial_num], aois_tokens_srcml, radius=25)
    
    assert aoi_fixes['trial'][0]==2
    assert aoi_fixes['participant'][0]=='6'
    assert aoi_fixes['code_file'][0]=='rectangle_java2.jpg'
    assert aoi_fixes['code_language'][0]=='rectangle_java2.jpg'
    assert aoi_fixes['timestamp'][0]==2768972532
    assert aoi_fixes['duration'][0]==96
    assert aoi_fixes['x_cord'][0]==677.545000
    assert aoi_fixes['y_cord'][0]==217.17458333333332
    assert aoi_fixes['aoi_x'][0]==657.5
    assert aoi_fixes['aoi_y'][0]==222
    assert aoi_fixes['aoi_width'][0]==52
    assert aoi_fixes['aoi_height'][0]==21
    assert aoi_fixes['token'][0]=='class'
    assert aoi_fixes['length'][0]==5
    assert aoi_fixes['srcML'][0]=='class'

    assert aoi_fixes['trial'][1]==2
    assert aoi_fixes['participant'][1]=='6'
    assert aoi_fixes['code_file'][1]=='rectangle_java2.jpg'
    assert aoi_fixes['code_language'][1]=='rectangle_java2.jpg'
    assert aoi_fixes['timestamp'][1]==2770019801
    assert aoi_fixes['duration'][1]==60
    assert aoi_fixes['x_cord'][1]==744.0753333333333
    assert aoi_fixes['y_cord'][1]==216.61266666666666
    assert aoi_fixes['aoi_x'][1]==715.5
    assert aoi_fixes['aoi_y'][1]==222
    assert aoi_fixes['aoi_width'][1]==90
    assert aoi_fixes['aoi_height'][1]==21
    assert aoi_fixes['token'][1]=='Rectangle'
    assert aoi_fixes['length'][1]==9
    assert aoi_fixes['srcML'][1]=='class->name'

def test_AlMadi_Dataset():
    '''Test reading data in .asc format'''
    AlMadiPath = tk.download("AlMadi2018")
    EMIP = tk.AlMadi_dataset('datasets/AlMadi2018/ASCII/', 8)   # gets the structured data of 8 subjects
    
    assert len(EMIP)==8
    assert EMIP['001'].trial[0].get_subject_id()=='datasets/AlMadi2018/ASCII/001'
    assert EMIP['001'].get_number_of_trials() == 16
    assert EMIP['001'].trial[0].get_sample_number()==25

    subject_ID = '001'
    trial_num = 1  
    
    assert EMIP[subject_ID].trial[trial_num].get_fixation_number()==14
    assert EMIP[subject_ID].trial[trial_num].get_sample_number()==29
    assert EMIP[subject_ID].trial[trial_num].get_trial_image()=="5667346413132987794.png"


def test_AlMadi_offset():
    #AlMadiPath = tk.download("AlMadi2018")
    EMIP = tk.AlMadi_dataset('datasets/AlMadi2018/ASCII/', 8)   # gets the structured data of 8 subjects
    subject_ID = '001'
    trial_num = 1  

    EMIP[subject_ID].trial[trial_num].sample_offset(-200, 100)
    assert EMIP[subject_ID].trial[trial_num].get_offset()==(-200, 100)
    
    EMIP[subject_ID].trial[trial_num].reset_offset()
    assert EMIP[subject_ID].trial[trial_num].get_offset()==(0, 0)
