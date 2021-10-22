class Fixation:
    """ Basic container for storing Fixation data """

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

    def __str__(self):
        """Returns string information of fixation

        Returns
        -------
        str
            fixation information
        """
        return f"{self.trial_id} {self.participant_id} {self.timestamp} {self.duration} " \
               f"{self.x_cord} {self.y_cord} {self.token} {self.pupil}"
