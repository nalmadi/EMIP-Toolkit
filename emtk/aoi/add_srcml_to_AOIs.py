import pandas as pd

def add_srcml_to_AOIs(aois_raw, srcML_path):
    """Adds srcML tags to AOIs dataframe and returns it.
        Check https://www.srcml.org/ for more information about srcML

        The files: rectangle.tsv and vehicle.tsv should be in the same directory as the code.

    Parameters
    ----------
    aois_raw : pandas.Dataframe
        the dataframe where AOIs are stored

    srcML_path : string
        the path of the srcML tags file

    Returns
    -------
    pandas.DataFrame
        AOI dataframe with srcML
    """

    image_name = aois_raw["image"][1]

    # srcML rectangle files
    if image_name == "rectangle_java.jpg":
        file_name = "rectangle.tsv"

    if image_name == "rectangle_java2.jpg":
        file_name = "rectangle.tsv"

    if image_name == "rectangle_python.jpg":
        aois_raw["srcML_tag"] = 'na'
        return aois_raw

    if image_name == "rectangle_scala.jpg":
        aois_raw["srcML_tag"] = 'na'
        return aois_raw

    # srcML vehicle files
    if image_name == "vehicle_java.jpg":
        file_name = "vehicle.tsv"

    if image_name == "vehicle_java2.jpg":
        file_name = "vehicle.tsv"

    if image_name == "vehicle_python.jpg":
        aois_raw["srcML_tag"] = 'na'
        return aois_raw

    if image_name == "vehicle_scala.jpg":
        aois_raw["srcML_tag"] = 'na'
        return aois_raw

    srcML_table = pd.read_csv(srcML_path + file_name, sep='\t')

    aois_raw = aois_raw[aois_raw.kind == "sub-line"].copy()

    # after the srcML file has been recognized
    # we can attach tokens to correct AOI

    tags = []

    for location in aois_raw["name"].iteritems():
        found = False

        for srcML_row in srcML_table.itertuples(index=True, name='Pandas'):
            # stimulus_file	token	AOI	syntactic_context

            if srcML_row.AOI == location[1]:
                tags.append(srcML_row.syntactic_context)
                found = True

        if not found:
            tags.append("na")

    aois_raw["srcML_tag"] = tags

    return aois_raw