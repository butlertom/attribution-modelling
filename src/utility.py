"""
Helper functions defined for use in the repository.
"""

import pandas as pd


def clean_data(df):
    """Cleans the data by correcting data types.

    Parameters
    ----------
    df: DataFrame
        the attribution data to be cleaned

    Returns
    -------
    DataFrame that has been cleaned
    """

    # rename time to datetime, as this is more appropriate
    df.rename(columns={"time": "datetime"}, inplace=True)

    # order the data by the cookie and datetime
    df.sort_values(["cookie", "datetime"], inplace=True)

    # convert the datetime column into the matching data type
    df["datetime"] = pd.to_datetime(df["datetime"])

    # get a path order to the data
    df["path_order"] = df.groupby("cookie").cumcount() + 1

    # separate out the date and time components from the datetime feature
    df["date"] = df["datetime"].dt.date.astype("datetime64[ns]")
    df["time"] = df["datetime"].dt.time

    # create dummies from the channel column
    df = pd.merge(
        df,
        df["channel"].str.get_dummies(),
        left_index=True,
        right_index=True,
    )

    return df


def summarise_paths(df):
    """Summarises the data to a cookie level, including the path/journey of
    interactions the user took

    Parameters
    ----------
    df: DataFrame
        Cleaned data output from the clean_data function

    Returns
    -------
    DataFrame of summarised data
    """

    agg = df.groupby("cookie").agg(
        interactions=pd.NamedAgg("interaction", "count"),
        first_interaction=pd.NamedAgg("datetime", "min"),
        last_interaction=pd.NamedAgg("datetime", "max"),
        conversion=pd.NamedAgg("conversion", "max"),
        conversion_value=pd.NamedAgg("conversion_value", "sum"),
        channels_interacted_with=pd.NamedAgg("channel", "nunique"),
        facebook=pd.NamedAgg("Facebook", "sum"),
        instagrame=pd.NamedAgg("Instagram", "sum"),
        online_display=pd.NamedAgg("Online Display", "sum"),
        online_video=pd.NamedAgg("Online Video", "sum"),
        paid_search=pd.NamedAgg("Paid Search", "sum"),
    )
    agg["journey_length"] = agg["last_interaction"] - agg["first_interaction"]

    # now derive a path, separated by '>', of channels interacted with
    paths = (
        df.groupby("cookie")["channel"].apply(list).apply(lambda lst: " > ".join(lst))
    )
    paths.rename("path", inplace=True)

    # join the path metrics and the path together
    df_agg = pd.merge(agg, paths, left_index=True, right_index=True)

    df_agg.reset_index(inplace=True)

    return df_agg
