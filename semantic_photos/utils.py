from typing import List, Dict, Any


def describe_people_in_scene(people: List[str]) -> str:
    if len(people) == 1:
        return f"The scene contains {people[0]}."

    return f"The scene contains {', '.join(people[:-1])} and {people[-1]}."


def describe_geo_location(geos: List[Dict[str, Any]]):
    names = []
    for g in geos:
        print(g)
        names.append(f"{g['toponymName']}, {g['adminName1']}")

    if len(names) == 1:
        return f"The scene takes place in {names[0]}."
    return f"The scene contains {', '.join(names[:-1])} and {names[-1]}."
