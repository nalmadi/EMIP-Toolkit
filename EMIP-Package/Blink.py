class Blink:
    def __init__(self, trial_id, participant_id, timestamp, duration):
        """Initializes the basic data for each blink

        Parameters
        ----------
        trial_id : int
            trial id that the blink belongs to

        participant_id : str
            participant id that the blink belongs to

        timestamp : int
            blink time stamp

        duration : int
            blink duration in milliseconds
        """
        self.trial_id = trial_id
        self.participant_id = participant_id
        self.timestamp = timestamp
        self.duration = duration

    def get_blink(self):
        """Returns blink attributes as a list

        Returns
        -------
        list
            a list containing blink attributes
        """

        return [self.trial_id,
                self.participant_id,
                self.timestamp,
                self.duration]

    def __str__(self):
        """Returns string information of blink

        Returns
        -------
        str
            blink information
        """
        return f"{self.trial_id} {self.participant_id} {self.timestamp} {self.duration}"
