'''
John, Henry, Ryan
Pytest for EMIP Toolkit
CS321 Final Project
December 7, 2021
'''
import emip_toolkit as tk

def test_EMIP_dataset():
    '''Testing reading raw files from EMIP Dataset'''
    
    EMIP = tk.EMIP_dataset('./emip_dataset/testdata/', 10)
    assert len(EMIP)==10
    assert EMIP['100'].trial[0].get_subject_id()=='100'
    assert EMIP['100'].get_number_of_trials()==7
    assert EMIP['100'].trial[0].get_sample_number()==12040


def test_fixation_filter():
    '''Tests getting a specific trial and subject and number of fixations'''
    
    EMIP = tk.EMIP_dataset('./emip_dataset/testdata/', 10)
    subject_ID = '106'
    trial_num = 2 

    assert EMIP[subject_ID].trial[trial_num].get_fixation_number() == 357   
    assert EMIP[subject_ID].trial[trial_num].get_sample_number() == 18964
    assert EMIP[subject_ID].trial[trial_num].get_trial_image() == "vehicle_java2.jpg"
    
    
def test_offset():
    '''Testing the offset functionality'''

    EMIP = tk.EMIP_dataset('./emip_dataset/testdata/', 10)
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


def test_hit_test():
    '''Tests the hit test between fixation and aois'''

    EMIP = tk.EMIP_dataset('./emip_dataset/testdata/', 10)

    subject_ID = '106'
    trial_num = 2 

    image_path = "./emip_dataset/stimuli/"
    image = "rectangle_java2.jpg"
    aoi = tk.find_aoi(image, image_path, level="sub-line")
    file_path = "./emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)
    srcML_path = "./datasets/EMIP2021/"
    
    aois_tokens_srcml = tk.add_srcml_to_AOIs(aois_with_tokens, srcML_path)
    aoi_fixes = tk.hit_test(EMIP[subject_ID].trial[trial_num], aois_tokens_srcml, radius=25)
    
    assert aoi_fixes['trial'][0]==2
    assert aoi_fixes['participant'][0]=='106'
    assert aoi_fixes['code_file'][0]=='rectangle_java2.jpg'
    assert aoi_fixes['code_language'][0]=='rectangle_java2.jpg'
    assert aoi_fixes['timestamp'][0]==21891003504
    assert aoi_fixes['duration'][0]==68
    assert aoi_fixes['x_cord'][0]==807.9623529411765
    assert aoi_fixes['y_cord'][0]==367.9970588235294
    assert aoi_fixes['aoi_x'][0]==806.5
    assert aoi_fixes['aoi_y'][0]==367
    assert aoi_fixes['aoi_width'][0]==15
    assert aoi_fixes['aoi_height'][0]==21
    assert aoi_fixes['token'][0]=='='
    assert aoi_fixes['length'][0]==1
    assert aoi_fixes['srcML'][0]=='class->block->constructor->block->block_content->expr_stmt->expr->operator'

    assert aoi_fixes['trial'][1]==2
    assert aoi_fixes['participant'][1]=='106'
    assert aoi_fixes['code_file'][1]=='rectangle_java2.jpg'
    assert aoi_fixes['code_language'][1]=='rectangle_java2.jpg'
    assert aoi_fixes['timestamp'][1]==21892223446
    assert aoi_fixes['duration'][1]==60
    assert aoi_fixes['x_cord'][1]==792.2586666666666
    assert aoi_fixes['y_cord'][1]==326.1046666666667
    assert aoi_fixes['aoi_x'][1]==730.5
    assert aoi_fixes['aoi_y'][1]==331
    assert aoi_fixes['aoi_width'][1]==71
    assert aoi_fixes['aoi_height'][1]==19
    assert aoi_fixes['token'][1]=='this.x1'
    assert aoi_fixes['length'][1]==7
    assert aoi_fixes['srcML'][1]=='class->block->constructor->block->block_content->expr_stmt->expr->name->name->operator->name'

def test_AlMadi_Dataset():
    '''Test reading data in .asc format'''
    EMIP = tk.AlMadi_dataset('ASCII/', 8)   # gets the structured data of 8 subjects
    
    assert len(EMIP)==8
    assert EMIP['001'].trial[0].get_subject_id()=='ASCII/001'
    assert EMIP['001'].get_number_of_trials() == 16
    assert EMIP['001'].trial[0].get_sample_number()==25

    subject_ID = '001'
    trial_num = 1  
    
    assert EMIP[subject_ID].trial[trial_num].get_fixation_number()==14
    assert EMIP[subject_ID].trial[trial_num].get_sample_number()==29
    assert EMIP[subject_ID].trial[trial_num].get_trial_image()=="5667346413132987794.png"


def test_AlMadi_offset():
    EMIP = tk.AlMadi_dataset('ASCII/', 8)   # gets the structured data of 8 subjects
    subject_ID = '001'
    trial_num = 1  

    EMIP[subject_ID].trial[trial_num].sample_offset(-200, 100)
    assert EMIP[subject_ID].trial[trial_num].get_offset()==(-200, 100)
    
    EMIP[subject_ID].trial[trial_num].reset_offset()
    assert EMIP[subject_ID].trial[trial_num].get_offset()==(0, 0)