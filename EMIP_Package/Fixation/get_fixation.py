def get_fixation(self):
    """Returns fixation attributes as a list

    Returns
    -------
    list
        a list containing fixation information
    """

    return [attribute for attribute in

            [self.trial_id,
             self.participant_id,
             self.timestamp,
             self.duration,
             self.x_cord,
             self.y_cord,
             self.token,
             self.pupil]

            if attribute is not None]
