from enum import Enum


class Supported(Enum):
    MACOS_PHOTO_LIBRARY = "macos_photo_library"
    DIGIKAM_PHOTO_LIBRARY = "digikam_photo_library"

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return str(self)
    
    @staticmethod
    def argparse(choice: str):
        try:
            return Supported[choice.upper()]
        except KeyError:
            return choice
