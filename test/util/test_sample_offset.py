import numpy as np
import pytest

from emtk.parsers import EMIP, AlMadi
from emtk.util import sample_offset

def test_sample_offset_with_EMIP():
    eye_events, samples = EMIP(1)

    offset_eye_events = sample_offset(eye_events, -200, 100)

    x0_offset = ( offset_eye_events["x0"] - eye_events["x0"] ).unique()
    x0_offset_np = np.array(x0_offset)
    assert pytest.approx( x0_offset_np, 0.01 ) == -200

    y0_offset = ( offset_eye_events["y0"] - eye_events["y0"] ).unique()
    y0_offset_np = np.array(y0_offset)
    assert pytest.approx( y0_offset_np, 0.01 ) == 100

    x1_offset = ( offset_eye_events["x1"] - eye_events["x1"] ).unique()
    assert np.isnan(x1_offset)

    y1_offset = ( offset_eye_events["y1"] - eye_events["y1"] ).unique()
    assert np.isnan(y1_offset)


    offset_samples = sample_offset(samples, 148.45, -28324.1224, 
                                      x0_col = 'L POR X [px]', 
                                      y0_col = 'L POR Y [px]')

    sample_x0_offset = ( offset_samples['L POR X [px]'] - samples['L POR X [px]'] ).unique()
    sample_x0_offset_np = np.array( sample_x0_offset )
    assert pytest.approx( sample_x0_offset_np, 0.01 ) == 148.45

    sample_y0_offset = ( offset_samples['L POR Y [px]'] - samples['L POR Y [px]'] ).unique()
    sample_y0_offset_np = np.array( sample_y0_offset )
    assert pytest.approx( sample_y0_offset_np, 0.01 ) == -28324.1224



def test_sample_offset_with_AlMadi():
    eye_events, _ = AlMadi(8)

    offset_eye_events = sample_offset(eye_events, -200, 100)

    x0_offset = ( offset_eye_events["x0"] - eye_events["x0"] ).unique()
    x0_offset_np = np.array(x0_offset)
    assert np.isnan( np.sum(x0_offset_np) )
    assert pytest.approx( x0_offset_np[~np.isnan(x0_offset_np)] ) == -200

    y0_offset = ( offset_eye_events["y0"] - eye_events["y0"] ).unique()
    y0_offset_np = np.array(y0_offset)
    assert np.isnan( np.sum(y0_offset_np) )
    assert pytest.approx( y0_offset_np[~np.isnan(y0_offset_np)] ) == 100

    x1_offset = ( offset_eye_events["x1"] - eye_events["x1"] ).unique()
    x1_offset_np = np.array(x1_offset)
    assert np.isnan( np.sum(x1_offset_np) )
    assert pytest.approx( x1_offset_np[~np.isnan(x1_offset_np)] ) == -200

    y1_offset = ( offset_eye_events["y1"] - eye_events["y1"] ).unique()
    y1_offset_np = np.array(y1_offset)
    assert np.isnan( np.sum(y1_offset_np) )
    assert pytest.approx( y1_offset_np[~np.isnan(y1_offset_np)] ) == 100
