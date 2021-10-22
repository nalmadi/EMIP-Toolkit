class Saccade:
    def __init__(self, trial_id, participant_id, timestamp, duration, x_cord, y_cord, x1_cord, y1_cord, amplitude,
                 peak_velocity):
        """Initializes the basic data for each fixation

        Parameters
        ----------
        trial_id : int
            trial id that the fixation belongs to

        participant_id : str
            participant id that the fixation belongs to

        timestamp : int
            saccade start time stamp

        duration : int
            saccade duration in milliseconds

        x_cord : float
            saccade start point x coordinate

        y_cord : float
            saccade start point y coordinate

        x1_cord : float
            saccade end point x coordinate

        y1_cord : float
            saccade end point y coordinate

        amplitude : float
            amplitude for saccade

        peak_velocity : int
            peak velocity during saccade
        """

        self.trial_id = trial_id
        self.participant_id = participant_id
        self.timestamp = timestamp
        self.duration = duration
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.x1_cord = x1_cord
        self.y1_cord = y1_cord
        self.amplitude = amplitude
        self.peak_velocity = peak_velocity

    def get_saccade(self):
        """Returns saccade attributes as a list

        Returns
        -------
        list
            a list containing saccade attributes
        """

        return [self.trial_id,
                self.participant_id,
                self.timestamp,
                self.duration,
                self.x_cord,
                self.y_cord,
                self.x1_cord,
                self.y1_cord,
                self.amplitude,
                self.peak_velocity]

    def sample_offset(self, x_offset, y_offset):
        """Returns the x and y coordinate of the saccade

        Parameters
        ----------
        x_offset : float
            offset to be applied on all fixations in the x-axis

        y_offset : float
            offset to be applied on all fixations in the y-axis
        """
        self.x_cord += x_offset
        self.x1_cord += x_offset
        self.y_cord += y_offset
        self.y1_cord += y_offset

    def __str__(self):
        """Returns string information of saccade

        Returns
        -------
        str
            saccade information
        """
        return f"{self.trial_id} {self.participant_id} {self.timestamp} {self.duration}" \
               f"{self.x_cord} {self.y_cord} {self.x1_cord} {self.y1_cord} {self.amplitude} {self.peak_velocity}"
