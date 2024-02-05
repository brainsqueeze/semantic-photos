from typing import Dict, Iterator, Any
from datetime import datetime
import warnings
import sqlite3
import sys
import os
import re

Cursor = sqlite3.Cursor
Row = sqlite3.Row


class ReaderBase:
    """Base reader class for photo libraries.
    """
    ALLOWED_TYPES = frozenset([
        "jpg",
        "png"
    ])

    @staticmethod
    def _dict_factory(cursor: Cursor, row: Row) -> Dict[Any, Any]:
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def file_count(self, dir_path: str) -> int:
        return sum(
            1 for _ in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, _))
            and _.split('.')[-1].lower() in self.ALLOWED_TYPES
        )

    @property
    def albums(self):
        pass


class DigikamReader(ReaderBase):
    """Reader class for Digikam photo library databases. This connects to the core database as well as the recognition
    database. Currently this is only supported on Linux operating systems, with cross-platform support coming in the
    future.

    Parameters
    ----------
    path : str
        Absolute path to the parent directory of the database files.
    core_db : str, optional
        Sqlite filename for the principle database, by default "digikam4.db"
    recognition_db : str, optional
        Sqlite filename for the recognition database, by default "recognition.db"
    """

    UUID_PATTERN = re.compile(r"^(volumeid:\?)(uuid=)(.*)")

    def __init__(
        self,
        path: str,
        core_db: str = "digikam4.db",
        recognition_db: str = "recognition.db"
    ):
        self.__connection = sqlite3.connect(database=os.path.join(path, core_db))
        self.__connection.row_factory = self._dict_factory
        self.__cursor = self.__connection.cursor()

        self.__cursor.execute("ATTACH DATABASE ? as recog;", (os.path.join(path, recognition_db),))
        self.__connection.commit()

        self.__volume_map = {}
        for row in self.__cursor.execute("SELECT id, identifier, specificPath FROM AlbumRoots;"):
            if (m := self.UUID_PATTERN.match(row["identifier"])):
                volume_uuid = m.group(3)

                if sys.platform == "linux":
                    root = os.path.join("/mnt", volume_uuid.upper())
                else:
                    warnings.warn(f"Platform `{sys.platform}` is not yet supported")
                    continue

                specific_path: str = row["specificPath"]
                if specific_path.startswith('/'):
                    specific_path = specific_path[1:]
                self.__volume_map[row["id"]] = {
                    "root": root,
                    "relative": specific_path
                }

    def stream_media_from_album(self, album_id: int) -> Iterator[Dict[str, Any]]:
        query = """
        SELECT alb.id AS album_id
        , img.id AS image_id
        , alb.relativePath
        , img.name
        , info.creationDate AS creation_date
        , pos.latitudeNumber AS latitude
        , pos.longitudeNumber AS longitude
        , people.people_names
        FROM Images AS img
        INNER JOIN Albums AS alb ON img.album = alb.id
        INNER JOIN ImageInformation AS info ON img.id = info.imageid
        LEFT JOIN ImagePositions AS pos ON img.id = pos.imageid
        LEFT JOIN (
            SELECT itp.imageid
            , GROUP_CONCAT(tp.value) AS people_names
            FROM ImageTagProperties AS itp
            INNER JOIN TagProperties AS tp ON itp.tagid = tp.tagid
            WHERE tp.property = 'faceEngineId'
            GROUP BY itp.imageid
        ) AS people ON img.id = people.imageid
        WHERE alb.id = ?;
        """

        self.__cursor.execute(query, (album_id,))
        for row in self.__cursor:
            relative_path: str = row["relativePath"]
            if relative_path.startswith('/'):
                relative_path = relative_path[1:]

            row["relativePath"] = relative_path
            row["creation_date"] = datetime.strptime(row["creation_date"], '%Y-%m-%dT%H:%M:%S.%f')
            yield row

    @property
    def albums(self) -> Dict[str, Dict[str, Any]]:
        query = """
        SELECT id
        , albumRoot
        , relativePath
        FROM Albums;
        """

        output = {}
        for row in self.__cursor.execute(query):
            relative_path: str = row["relativePath"]
            if relative_path.startswith('/'):
                relative_path = relative_path[1:]

            album_root = self.__volume_map[row["albumRoot"]]
            path = os.path.join(album_root["root"], album_root["relative"], relative_path)
            output[relative_path] = {
                "album_id": row["id"],
                "name": relative_path.replace('/', ' :: '),
                "path": path,
                "count": self.file_count(path)
            }
        return output

    @property
    def count(self):
        return self.__cursor.rowcount

    @property
    def volume_map(self):
        return self.__volume_map

    def teardown(self):
        self.__cursor.close()
        self.__connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.teardown()
