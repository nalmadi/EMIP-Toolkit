def sample_offset(self, x_offset, y_offset):
    """Returns the x and y coordinate of the fixation

    Parameters
    ----------
    x_offset : float
        offset to be applied on all fixations in the x-axis

    y_offset : float
        offset to be applied on all fixations in the y-axis
    """
    self.x_cord += x_offset
    self.y_cord += y_offset
