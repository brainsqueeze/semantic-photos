from typing import Any


def describe_people_in_scene(people: list[str]) -> str:
    """Builds a string that describes the named people identified in a given photo.

    Parameters
    ----------
    people : list[str]
        List of people names

    Returns
    -------
    str
    """

    if not people:
        return ""
    if len(people) == 1:
        return f"The scene contains {people[0]}."

    return f"The scene contains {', '.join(people[:-1])} and {people[-1]}."


def describe_geo_location(geos: list[dict[str, Any]]) -> str:
    """Builds a string that describes the geographic location where the photo was taken.
    Can contain place names or points of interest.

    Parameters
    ----------
    geos : list[dict[str, Any]]
        Geographic information from the Geonames API

    Returns
    -------
    str
    """

    if not geos:
        return ""
    names = []
    for g in geos:
        names.append(f"{g['toponymName']}, {g['adminName2']}, {g['adminName1']}")

    if len(names) == 1:
        return f"The scene takes place in {names[0]}."
    return f"The scene takes place in {', '.join(names[:-1])} and {names[-1]}."
