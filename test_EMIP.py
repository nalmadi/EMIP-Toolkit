import emip_toolkit as tk

data_path = tk.download("EMIP")
EMIP = tk.EMIP_dataset(data_path + '/EMIP-Toolkit- replication package/emip_dataset/rawdata/', 2)
subject_ID = '101'
trial_num = 2
EMIP[subject_ID].trial[trial_num].sample_offset(-200, 100)
image_path = data_path + '/EMIP-Toolkit- replication package/emip_dataset/stimuli/'
image = "rectangle_java2.jpg"
file_path = data_path + "/EMIP-Toolkit- replication package/emip_dataset/EMIP_DataCollection_Materials/emip_stimulus_programs/"
srcML_path = "./datasets/EMIP2021/"

def test_EMIP_dataset():

    assert len(EMIP) == 10
    assert EMIP['109'].trial[0].get_subject_id() == '109'
    assert EMIP['109'].get_number_of_trials() == 7
    assert EMIP['109'].trial[0].get_sample_number() == 3737

def test_get_fixation_number():

    assert EMIP[subject_ID].trial[trial_num].get_fixation_number() == 192
    

def test_get_sample_number():

    assert EMIP[subject_ID].trial[trial_num].get_sample_number() == 8593

def test_get_trial_image():

    assert EMIP[subject_ID].trial[trial_num].get_trial_image() == 'vehicle_python.jpg'

def test_trial_sample_offset():

    assert EMIP[subject_ID].trial[trial_num].get_offset() == (-200, 100)

def test_trial_reset_offset():

    EMIP[subject_ID].trial[trial_num].reset_offset()

    assert EMIP[subject_ID].trial[trial_num].get_offset() == (0,0)

def test_find_aoi():

    aoi = tk.find_aoi(image, image_path, level="sub-line")

    assert aoi.columns.values.tolist() == ['kind', 'name', 'x', 'y', 'width', 'height', 'image']
    assert aoi.iloc[0].values.tolist() == ['sub-line', 'line 1 part 1', 589.5, 222, 63, 21, 'rectangle_java2.jpg']

    aoi = tk.find_aoi(image, image_path, level="line")
    assert aoi.columns.values.tolist() == ['kind', 'name', 'x', 'y', 'width', 'height', 'image']
    assert aoi.iloc[0].values.tolist() == ['line', 'line 1', 589.5, 222, 234, 21, 'rectangle_java2.jpg']

def test_add_tokens_to_AOIs():

    aoi = tk.find_aoi(image, image_path, level="sub-line")
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)

    assert aois_with_tokens.columns.values.tolist() == ['kind', 'name', 'x', 'y', 'width', 'height', 'image', 'token']
    assert aois_with_tokens.iloc[0].values.tolist() == ['sub-line', 'line 1 part 1', 589.5, 222, 63, 21, 'rectangle_java2.jpg', 'public']

    aoi = tk.find_aoi(image, image_path, level="line")
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)

    assert aois_with_tokens.columns.values.tolist() == ['kind', 'name', 'x', 'y', 'width', 'height', 'image', 'token']
    assert aois_with_tokens.iloc[0].values.tolist() == ['line', 'line 1', 589.5, 222, 234, 21, 'rectangle_java2.jpg', 'public class Rectangle {']

def test_add_srcml_to_AOIs(aois_with_tokens, srcML_path):

    aoi = tk.find_aoi(image, image_path, level="sub-line")
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)
    aois_tokens_srcml = tk.add_srcml_to_AOIs(aois_with_tokens, srcML_path)

    assert aois_tokens_srcml.columns.values.tolist() == ['kind', 'name', 'x', 'y', 'width', 'height', 'image', 'token', 'srcML_tag']
    assert aois_tokens_srcml.iloc[0].values.tolist() == ['sub-line', 'line 1 part 1', 589.5, 222, 63, 21, 'rectangle_java2.jpg', 'public', 'class->specifier']

def test_hit_test():

    aoi = tk.find_aoi(image, image_path, level="sub-line")
    aois_with_tokens = tk.add_tokens_to_AOIs(file_path, aoi)
    aois_tokens_srcml = tk.add_srcml_to_AOIs(aois_with_tokens, srcML_path)
    aoi_fixes = tk.hit_test(EMIP[subject_ID].trial[trial_num], aois_tokens_srcml, radius=25)

    assert aoi_fixes.columns.values.tolist() == ['trial', 'participant', 'code_file', 'code_language', 'timestamp', 'duration', 'x_cord', 'y_cord', 'aoi_x', 'aoi_y', 'aoi_width', 'aoi_height', 'token', 'length', 'srcML']
    assert aoi_fixes.iloc[0].values.tolist() == [2, '101', 'rectangle_java2.jpg','rectangle_java2.jpg', 443075272, 80, 722.756, 376.2955, 730.5, 367, 71, 21, 'this.y1', 7, 'class->block->constructor->block->block_content->expr_stmt->expr->name->name->operator->name']
