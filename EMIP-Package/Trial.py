class Trial:
    """Each trial consists of many samples that need to be converted to fixations.
        A trial is part of an experiment. Or each experiment consists of multiple trials.
    """

    def __init__(self, trial_id: int, participant_id: str, image: str, fixations: dict, saccades: dict, blinks: dict,
                 samples: list, eye_tracker: str):
        """Initializes attributes for storing trial data, fixations, saccades, blinks, and
        stores image name

        Parameters
        ----------
        trial_id : int
            id of this trial

        participant_id : str
            id of this participant

        image : str
            image path for this trial

        fixations : dict
            dictionary that stores fixations as values, order of eye movement in the trial as key

        saccades : dict
            dictionary that stores saccades as values, order of eye movement in the trial as key

        blinks : dict
            dictionary that stores blinks as values, order of eye movement in the trial as key

        samples : list
            list of raw data samples

        eye_tracker : str
            type of eye tracker
        """
        self.trial_id = trial_id
        self.participant_id = participant_id
        self.image = image
        self.fixations = fixations
        self.saccades = saccades
        self.blinks = blinks
        self.samples = samples

        self.offset_history = [[0, 0]]

        self.eye_tracker = eye_tracker

    def get_trial_id(self):
        """Returns the trial id

        Returns
        -------
        int
            trial id
        """
        return self.trial_id

    def get_subject_id(self):
        """Returns the participant id

        Returns
        -------
        str
            participant id
        """
        return self.participant_id

    def get_trial_image(self):
        """Returns the image filename associated with the trial

        Returns
        -------
        str
            the image filename associated with the trial
        """
        return self.image

    def get_fixations(self):
        """Returns the fixations in the trial

        Returns
        -------
        dict
            fixations in the trial
        """
        return self.fixations

    def get_fixation_number(self):
        """Returns the number of fixations in the trial

        Returns
        -------
        int
            number of fixations in the trial
        """
        return len(self.fixations)

    def get_saccades(self):
        """Returns the saccades in the trial

        Returns
        -------
        dict
            saccades in the trial
        """
        return self.saccades

    def get_saccade_number(self):
        """Returns the number of saccades in the trial

        Returns
        -------
        int
            number of saccades in the trial
        """
        return len(self.saccades)

    def get_blinks(self):
        """Returns the blinks in the trial

        Returns
        -------
        dict
            blinks in the trial
        """
        return self.blinks

    def get_blink_number(self):
        """Returns the number of blinks in the trial

        Returns
        -------
        int
            number of blinks in the trial
        """
        return len(self.blinks)

    def get_eye_movement_number(self):
        """Returns the total number of eye movement in the trial

        Returns
        -------
        int
            total number of eye movement
        """
        return self.get_fixation_number() + self.get_saccade_number() + self.get_blink_number()

    def get_samples(self):
        """Returns the raw sample in a list

        Returns
        -------
        list
            a list of raw eye movement samples
        """
        return self.samples

    def get_sample_number(self):
        """Returns the total number of eye movement in the trial

        Returns
        -------
        int
            total number of eye movement
        """
        return len(self.samples)

    def get_offset(self):
        """Returns total offset applied by adding all offsets in offset history

        Returns
        -------
        tuple
            x_offset, y_offset
        """
        return tuple(np.array(self.offset_history).sum(axis=0))

    def reset_offset(self):
        """Resets and changes previously done using offset it implements UNDO feature by
            removing the all applied offset from the offset history.
        """

        x_total, y_total = tuple(np.array(self.offset_history).sum(axis=0) * -1)

        self.sample_offset(x_total, y_total)

        self.offset_history = [[0, 0]]

    def sample_offset(self, x_offset, y_offset):
        """Moves samples +X and +Y pixels across the viewing window to correct fixation shift or
            other shifting problems manually

        Parameters
        ----------
        x_offset : int
            offset to be applied on all fixations in the x-axis

        y_offset : int
            offset to be applied on all fixations in the y-axis
        """
        self.offset_history.append([x_offset, y_offset])

        for order in self.fixations.keys():
            self.fixations[order].sample_offset(x_offset, y_offset)

        for order in self.saccades.keys():
            self.saccades[order].sample_offset(x_offset, y_offset)

        # go over all samples (SMPs) in trial data
        for sample in self.samples:

            if self.eye_tracker == "SMIRed250":
                # Filter MSG samples if any exist, or R eye is inValid

                x_cord, y_cord = float(sample[23]), float(sample[24])

                sample[23] = str(x_cord + x_offset)
                sample[24] = str(y_cord + y_offset)

    def __draw_raw_data(self, draw):
        """Private method that draws raw sample data

        Parameters
        ----------
        draw : PIL.ImageDraw.Draw
            a Draw object imposed on the image
        """

        if self.eye_tracker == "SMIRed250":
            for sample in self.samples:
                # Invalid records
                if len(sample) > 5:
                    x_cord = float(sample[23])
                    y_cord = float(sample[24])  # - 150

                dot_size = 2

                draw.ellipse((x_cord - (dot_size / 2),
                              y_cord - (dot_size / 2),
                              x_cord + dot_size, y_cord + dot_size),
                             fill=(255, 0, 0, 100))

        elif self.eye_tracker == "EyeLink1000":
            return
        return None

    def __draw_fixation(self, draw, draw_number=False):
        """Private method that draws the fixation, also allow user to draw eye movement order

        Parameters
        ----------
        draw : PIL.ImageDraw.Draw
            a Draw object imposed on the image

        draw_number : bool
            whether user wants to draw the eye movement number
        """
        for count, fixation in self.fixations.items():
            duration = fixation.duration
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                r = 5 * (duration / 100)

            x = fixation.x_cord
            y = fixation.y_cord

            bound = (x - r, y - r, x + r, y + r)
            outline_color = (255, 255, 0, 0)
            fill_color = (242, 255, 0, 128)
            draw.ellipse(bound, fill=fill_color, outline=outline_color)

            if draw_number:
                text_bound = (x, y - r / 2)
                text_color = (255, 0, 0, 225)
                draw.text(text_bound, str(count + 2), fill=text_color)

        return None

    def __draw_aoi(self, draw, aoi, bg_color):
        """Private method to draw the Area of Interest on the image

        Parameters
        ----------
        draw : PIL.ImageDraw.Draw
            a Draw object imposed on the image

        aoi : pandas.DataFrame
            a DataFrame that contains the area of interest bounds

        bg_color : str
            background color
        """

        outline = {'white': '#000000', 'black': '#ffffff'}

        for row in aoi[['x', 'y', 'width', 'height']].iterrows():
            y_coordinate = row[1]['y']
            x_coordinate = row[1]['x']
            height = row[1]['height']
            width = row[1]['width']
            draw.rectangle([(x_coordinate, y_coordinate),
                            (x_coordinate + width - 1, y_coordinate + height - 1)],
                           outline=outline[bg_color])

        return None

    def __draw_saccade(self, draw, draw_number=False):
        """

        Parameters
        ----------
        draw : PIL.ImageDraw.Draw
            a Draw object imposed on the image

        draw_number : bool
            whether user wants to draw the eye movement number
        """
        for count, saccade in self.saccades.items():
            x0 = saccade.x_cord
            y0 = saccade.y_cord
            x1 = saccade.x1_cord
            y1 = saccade.y1_cord

            bound = (x0, y0, x1, y1)
            line_color = (122, 122, 0, 255)
            penwidth = 2
            draw.line(bound, fill=line_color, width=penwidth)

            font = ImageFont.truetype('Tohoma.ttf', 16)

            if draw_number:
                text_bound = ((x0 + x1) / 2, (y0 + y1) / 2)
                text_color = 'darkred'
                draw.text(text_bound, str(count + 2), font=font, fill=text_color)

    def draw_trial(self, image_path, draw_raw_data=False, draw_fixation=True, draw_saccade=False, draw_number=False,
                   draw_aoi=None, save_image=None):
        """Draws the trial image and raw-data/fixations over the image
            circle size indicates fixation duration

        image_path : str
            path for trial image file.

        draw_raw_data : bool, optional
            whether user wants raw data drawn.

        draw_fixation : bool, optional
            whether user wants filtered fixations drawn

        draw_saccade : bool, optional
            whether user wants saccades drawn

        draw_number : bool, optional
            whether user wants to draw eye movement number

        draw_aoi : pandas.DataFrame, optional
            Area of Interests

        save_image : str, optional
            path to save the image, image is saved to this path if it parameter exists
        """

        im = Image.open(image_path + self.image)

        if self.eye_tracker == "EyeLink1000":

            background_size = (1024, 768)
            background = Image.new('RGB', background_size, color='black')

            *_, width, _ = im.getbbox()
            # offset = int((1024 - width) / 2) - 10
            trial_location = (10, 375)

            background.paste(im, trial_location, im.convert('RGBA'))

            im = background.copy()


        bg_color = find_background_color(im.copy().convert('1'))

        draw = ImageDraw.Draw(im, 'RGBA')

        if draw_aoi and isinstance(draw_aoi, bool):
            aoi = find_aoi(image=self.image, img=im)
            self.__draw_aoi(draw, aoi, bg_color)

        if isinstance(draw_aoi, pd.DataFrame):
            self.__draw_aoi(draw, draw_aoi, bg_color)

        if draw_raw_data:
            self.__draw_raw_data(draw)

        if draw_fixation:
            self.__draw_fixation(draw, draw_number)

        if draw_saccade:
            self.__draw_saccade(draw, draw_number)

        plt.figure(figsize=(17, 15))
        plt.imshow(np.asarray(im), interpolation='nearest')

        if save_image is not None:
            # Save the image with applied offset

            image_name = save_image + \
                         str(self.participant_id) + \
                         "-t" + \
                         str(self.trial_id) + \
                         "-offsetx" + \
                         str(self.get_offset()[0]) + \
                         "y" + \
                         str(self.get_offset()[1]) + \
                         ".png"

            plt.savefig(image_name)

            print(image_name, "saved!")
