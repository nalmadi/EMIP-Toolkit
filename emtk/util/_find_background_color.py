from PIL import Image

def _find_background_color(stimuli_module: str = None, stimuli_name: str = None, 
                        image: Image = None) -> str:
    
    """Private function that identifies the background color of the image

    Parameters
    ----------
    image : PIL.Image
        a PIL (pillow fork) Image object

    Returns
    -------
    str
        the color of the background of the image
    """
    
    if image is None:
        if stimuli_module is None or stimuli_name is None:
            return
        image = Image.open(stimuli_module + stimuli_name).convert('1')
    else:
        image = image.convert('1')

    width, height = image.size

    color_result = []
    box_size = min(width, height) // 20

    # Move a tiny rectangle box to obtain most common color
    for x, y in zip(range(0, width, box_size), range(0, height, box_size)):
        box = (x, y, x + box_size, y + box_size)
        # image.crop(box).show()
        minimum, maximum = image.crop(box).getextrema()
        color_result.append(minimum)
        color_result.append(maximum)

    # Analyze and determine the background color
    if color_result.count(255) > color_result.count(0):
        bg_color = 'white'
    else:
        bg_color = 'black'

    return bg_color