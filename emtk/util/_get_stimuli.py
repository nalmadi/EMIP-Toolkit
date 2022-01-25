from PIL import Image
import os

def _get_stimuli(stimuli_module: str, stimuli_name: str, eye_tracker: str) -> Image:

    # Retrieve original image
    stimuli = Image.open( os.path.join( stimuli_module, stimuli_name ) )

    # Preprocessing
    if ( eye_tracker == "EyeLink1000" ):
        background = Image.new('RGB', (1024, 768), color='black')
        background.paste(stimuli, (100, 375), stimuli.convert('RGBA'))
        return background.copy()

    return stimuli



    