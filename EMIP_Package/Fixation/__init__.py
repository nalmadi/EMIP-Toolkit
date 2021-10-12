def __init__(self, trial_id, participant_id, timestamp, duration, x_cord, y_cord, token, pupil):
    """Initializes the basic data for each fixation

    Parameters
    ----------
    trial_id : int
        trial id that the fixation belongs to

    participant_id : str
        participant id that the fixation belongs to

    timestamp : int
        fixation time stamp

    duration : int
        fixation duration in milliseconds

    x_cord : float
        fixation x coordinates

    y_cord : float
        fixation y coordinates

    token : str
        the source code token which the fixation is on

    pupil : float
        pupil size of the fixation
    """

    self.trial_id = trial_id
    self.participant_id = participant_id
    self.timestamp = timestamp
    self.duration = duration
    self.x_cord = x_cord
    self.y_cord = y_cord
    self.token = token
    self.pupil = pupil
