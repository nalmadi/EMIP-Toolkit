from PIL import Image
import os

def _get_stimuli(stimuli_module: str, stimuli_name: str, eye_tracker: str) -> Image:
    '''Retrieve stimuli image.

    Parameters
    ----------
    stimuli_module : str
        Path to the directory of stimuli images.

    stimuli_name : str
        Name of stimuli image.

    eye_tracker : str
        Name of eye tracker.

    Returns
    -------
    Pillow.Image
        Stimuli image.
    '''

    # Retrieve original image
    stimuli = Image.open( os.path.join( stimuli_module, stimuli_name ) )

    # Preprocessing
    if ( eye_tracker == "EyeLink1000" ):
        background = Image.new('RGB', (1024, 768), color='black')
        background.paste(stimuli, (100, 375), stimuli.convert('RGBA'))
        return background.copy()

    return stimuli



    