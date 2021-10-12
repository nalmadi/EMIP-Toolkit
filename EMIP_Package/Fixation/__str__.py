def __str__(self):
    """Returns string information of fixation

    Returns
    -------
    str
        fixation information
    """
    return f"{self.trial_id} {self.participant_id} {self.timestamp} {self.duration} " \
           f"{self.x_cord} {self.y_cord} {self.token} {self.pupil}"
