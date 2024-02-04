from typing import Dict, Any
import warnings
import sqlite3
import sys
import os
import re

from elasticsearch import Elasticsearch


def _dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> Dict[Any, Any]:
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class ReaderBase:

    @staticmethod
    def file_count(dir_path: str) -> int:
        return sum(1 for _ in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, _)))


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
        self.__connection.row_factory = _dict_factory
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
    def volume_map(self):
        return self.__volume_map

    def teardown(self):
        self.__cursor.close()
        self.__connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.teardown()
