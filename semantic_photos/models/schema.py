from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ImageData:
    path: str
    album_name: str
    file_name: str
    created: datetime
    caption: str = field(default="")
    geo_description: str = field(default="")
    people_description: str = field(default="")

    @property
    def text(self) -> bool:
        texts = []
        for t in (self.caption, self.geo_description, self.people_description):
            if t.strip() != "":
                t = t.strip()
                if not t.endswith('.'):
                    t += "."
                texts.append(t)
        return ' '.join(texts)
