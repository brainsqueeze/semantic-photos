from typing import Dict, Iterator, Optional, Any
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from datetime import datetime
import warnings
import sqlite3
import shutil
import sys
import os
import re

Cursor = sqlite3.Cursor
Row = sqlite3.Row


@dataclass
class Media:
    """Media data from photo libraries
    """

    album_id: int
    image_id: int
    image_file_name: str
    creation_date: datetime
    relative_path: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    people_names: Optional[str] = None


class ReaderBase:
    """Base reader class for photo libraries.
    """
    ALLOWED_TYPES = frozenset([
        "jpg",
        "jpeg",
        "png",
        "heic"
    ])

    @staticmethod
    def _dict_factory(cursor: Cursor, row: Row) -> Dict[Any, Any]:
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def file_count(self, dir_path: str) -> int:
        """Count the number of files that are of the allowed types

        Parameters
        ----------
        dir_path : str
            Album path

        Returns
        -------
        int
            Count of allowed media files
        """

        return sum(
            1 for _ in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, _))
            and _.split('.')[-1].lower() in self.ALLOWED_TYPES
        )

    @property
    def albums(self):
        """Lookup of supported albums in the photo library
        """


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

                root = None
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

    def stream_media_from_album(self, album_id: int) -> Iterator[Media]:
        """Stream media files and metadata from the specified albumn

        Parameters
        ----------
        album_id : int

        Yields
        ------
        Iterator[Media]
        """

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
            yield Media(
                album_id=row["album_id"],
                image_id=row["image_id"],
                image_file_name=row["name"],
                relative_path=relative_path,
                creation_date=datetime.strptime(row["creation_date"], '%Y-%m-%dT%H:%M:%S.%f'),
                lat=row["latitude"],
                lon=row["longitude"],
                people_names=row["people_names"]
            )

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
    def volume_map(self) -> Dict[Any, Any]:
        """Disk volume info relevant for Digikam databases

        Returns
        -------
        Dict[Any, Any]
        """

        return self.__volume_map

    def teardown(self):
        """Close all SQLite connections and cursor objects
        """

        self.__cursor.close()
        self.__connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.teardown()


class MacPhotosReader(ReaderBase):
    """Reader class for MacOS photo library databases.

    Parameters
    ----------
    photolibrary_path : str
        Path to the MacOS photo library
    core_db : str, optional
        Name of the SQLite database to use, by default "Photos.sqlite"
    """

    def __init__(
        self,
        photolibrary_path: str,
        core_db: str = "Photos.sqlite",
    ):
        if photolibrary_path.startswith('~'):
            photolibrary_path = os.path.expanduser(photolibrary_path)

        # copy database to a temporary directory, connect to it and then do a back up to an in-memory
        with TemporaryDirectory(prefix="semanticphotos_osx_") as tmpdir:
            try:
                shutil.copy(
                    src=os.path.join(photolibrary_path, "database", core_db),
                    dst=os.path.join(tmpdir, core_db)
                )
                source = sqlite3.connect(database=os.path.join(tmpdir, core_db))
                self.__connection = sqlite3.connect(':memory:')
                source.backup(self.__connection)
            except Exception as ex:
                print(ex)
        self.__connection.row_factory = self._dict_factory
        self.__cursor = self.__connection.cursor()

        self.__relative_file_path = os.path.join(photolibrary_path, "originals")

    @property
    def albums(self) -> Dict[str, Dict[str, Any]]:
        query = f"""
        SELECT ZMOMENTLIST.ZSORTINDEX AS yearmonth
        , COUNT(*) AS size
        FROM ZASSET
        INNER JOIN ZMOMENT ON ZASSET.ZMOMENT = ZMOMENT.Z_PK
        INNER JOIN ZMOMENTLIST ON ZMOMENT.ZMEGAMOMENTLIST = ZMOMENTLIST.Z_PK
        WHERE ZASSET.ZTRASHEDSTATE = 0
        AND LOWER(SUBSTR(
            ZASSET.ZFILENAME,
            INSTR(ZASSET.ZFILENAME, '.') + 1,
            LENGTH(ZASSET.ZFILENAME) - INSTR(ZASSET.ZFILENAME, '.')
        )) IN ({', '.join(['?'] * len(self.ALLOWED_TYPES))})
        GROUP BY ZMOMENTLIST.ZSORTINDEX;"""

        output = {}
        self.__cursor.execute(query, (*self.ALLOWED_TYPES,))
        for row in self.__cursor:
            album = str(row["yearmonth"])
            output[album] = {
                "album_id": row["yearmonth"],
                "name": f"{album[:4]}-{album[4:]}",
                "path": None,
                "count": row["size"]
            }
        return output

    def stream_media_from_album(self, album_id: str) -> Iterator[Media]:
        """Stream media files and metadata from the specified albumn

        Parameters
        ----------
        album_id : int

        Yields
        ------
        Iterator[Media]
        """

        query = f"""
        SELECT ZASSET.ZFILENAME AS name
        , ZASSET.ZDIRECTORY AS relative_dir
        , ZASSET.ZLATITUDE AS latitude
        , ZASSET.ZLONGITUDE AS longitude
        , ZASSET.ZDATECREATED + 978307200 AS creation_date
        , people.people_names
        , ZMOMENTLIST.ZSORTINDEX AS yearmonth
        FROM ZASSET
        LEFT JOIN (
            SELECT ZDETECTEDFACE.ZASSET
            , GROUP_CONCAT(ZPERSON.ZFULLNAME) AS people_names
            FROM ZDETECTEDFACE
            INNER JOIN ZPERSON ON ZPERSON.Z_PK = ZDETECTEDFACE.ZPERSON
            WHERE ZPERSON.ZFULLNAME IS NOT NULL
            AND ZPERSON.ZFULLNAME <> ''
            GROUP BY ZDETECTEDFACE.ZASSET
        ) AS people ON ZASSET.Z_PK = people.ZASSET
        INNER JOIN ZMOMENT ON ZASSET.ZMOMENT = ZMOMENT.Z_PK
        INNER JOIN ZMOMENTLIST ON ZMOMENT.ZMEGAMOMENTLIST = ZMOMENTLIST.Z_PK
        WHERE ZASSET.ZTRASHEDSTATE = 0
        AND LOWER(SUBSTR(
            ZASSET.ZFILENAME,
            INSTR(ZASSET.ZFILENAME, '.') + 1,
            LENGTH(ZASSET.ZFILENAME) - INSTR(ZASSET.ZFILENAME, '.')
        )) IN ({', '.join(['?'] * len(self.ALLOWED_TYPES))})
        AND ZMOMENTLIST.ZSORTINDEX = ?;"""

        self.__cursor.execute(query, (*self.ALLOWED_TYPES, album_id,))
        for row in self.__cursor:
            if row["name"].split('.')[-1] in self.ALLOWED_TYPES:
                relative_path: str = row["relative_dir"]
                if relative_path.startswith('/'):
                    relative_path = relative_path[1:]

                relative_path = os.path.join(self.__relative_file_path, relative_path)

                yield Media(
                    album_id=row["yearmonth"],
                    image_id=None,
                    image_file_name=row["name"],
                    relative_path=relative_path,
                    creation_date=datetime.utcfromtimestamp(row["creation_date"]),
                    lat=row["latitude"],
                    lon=row["longitude"],
                    people_names=row["people_names"]
                )

    @property
    def count(self):
        return self.__cursor.rowcount

    def teardown(self):
        """Close all SQLite connections and cursor objects
        """

        self.__cursor.close()
        self.__connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.teardown()
