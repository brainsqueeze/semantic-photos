from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ImageData:
    """Dataclass for image object to be used for vector indexing.
    """

    path: str
    album_name: str
    file_name: str
    created: datetime
    caption: str = field(default="")
    geo_description: str = field(default="")
    people_description: str = field(default="")

    @property
    def text(self) -> str:
        """Text description builder. Concatenates all available descriptions including caption, people, and geographies.

        Returns
        -------
        str
        """

        texts = []
        for t in (self.caption, self.geo_description, self.people_description):
            if t.strip() != "":
                t = t.strip()
                if not t.endswith('.'):
                    t += "."
                texts.append(t)
        return ' '.join(texts)
