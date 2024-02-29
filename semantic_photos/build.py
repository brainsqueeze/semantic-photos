from typing import List, Dict, Tuple, Iterator, Any
import argparse
import warnings
import os

import torch
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC = True
except ImportError:
    HEIC = False
    warnings.warn("pillow-heif is not installed, HEIC images will be skipped")

from semantic_photos.galleries.database import (
    DigikamReader,
    MacPhotosReader,
    Media
)
from semantic_photos.geographies import GeonamesReverseGeocoder
from semantic_photos.models.caption import ImageCaption
from semantic_photos.models.documents import ImageVectorStore
from semantic_photos.models.schema import ImageData
from semantic_photos.utils import describe_people_in_scene, describe_geo_location
from semantic_photos.constants import Supported


def batch_caption(images: List[ImageData], captioner: ImageCaption) -> List[ImageData]:
    """Batch process image-to-text captioning.

    Parameters
    ----------
    images : List[ImageData]
    captioner : ImageCaption

    Returns
    -------
    List[ImageData]
        List of image data objects with updated caption text
    """

    captions = captioner.caption([img.path for img in images])

    for img, caption in zip(images, captions):
        img.caption = caption
    return images


def generate_geo_descriptions(image: ImageData, metadata: Media, geocoder: GeonamesReverseGeocoder) -> ImageData:
    """Reverse geo-code image location tags and generate a text description.

    Parameters
    ----------
    image : ImageData
    metadata : Media
        Metadata object from the photo library DB
    geocoder : GeonamesReverseGeocoder

    Returns
    -------
    ImageData
        Image data objects with updated geo description text
    """

    if metadata.lat and metadata.lon:
        geos = geocoder.find_nearby(
            latitude=metadata.lat,
            longitude=metadata.lon
        )
        image.geo_description = describe_geo_location(geos.get("geonames", []))
    return image


def generate_people_in_scene_descriptions(image: ImageData, metadata: Media) -> ImageData:
    """Generate a people-in-scene text description.

    Parameters
    ----------
    image : ImageData
    metadata : Media
        Metadata object from the photo library DB

    Returns
    -------
    ImageData
    """

    if metadata.people_names:
        image.people_description = describe_people_in_scene(metadata.people_names.split(','))
    return image


def stream_digikam_albums(
    photo_library_dir: str,
    albums: List[str]
) -> Iterator[Tuple[ImageData, Media]]:
    """Stream wrapper for the Digikam-based photolibrary reader.

    Parameters
    ----------
    photo_library_dir : str
        Absolute path to the directory containing the SQLite data.
    albums : List[str]
        Albums to process

    Yields
    ------
    Iterator[Tuple[ImageData, Media]]
        (Image object, metadata object)
    """

    with DigikamReader(photolibrary_path=photo_library_dir) as db:
        album_map = db.albums
        for album in albums:
            for record in tqdm(
                db.stream_media_from_album(album_id=album_map[album]["album_id"]),
                total=album_map[album]["count"],
                desc=f"Loading {album}"
            ):
                if record.image_file_name.lower().endswith('.heic') and not HEIC:
                    continue

                meta = album_map[record.relative_path]
                img_data = ImageData(
                    path=os.path.join(meta["path"], record.image_file_name),
                    album_name=meta["name"],
                    file_name=record.image_file_name,
                    created=record.creation_date,
                )
                yield img_data, record


def stream_macos_albums(
    photo_library_dir: str,
    albums: List[str]
) -> Iterator[Tuple[ImageData, Media]]:
    """Stream wrapper for the MacOS-based photolibrary reader.

    Parameters
    ----------
    photo_library_dir : str
        Absolute path to the directory containing the SQLite data.
    albums : List[str]
        Albums to process

    Yields
    ------
    Iterator[Tuple[ImageData, Media]]
        (Image object, metadata object)
    """

    with MacPhotosReader(photolibrary_path=photo_library_dir) as db:
        album_map = db.albums
        for album in albums:
            for record in tqdm(
                db.stream_media_from_album(album_id=album_map[album]["album_id"]),
                total=album_map[album]["count"],
                desc=f"Loading {album}"
            ):
                if record.image_file_name.lower().endswith('.heic') and not HEIC:
                    continue

                img_data = ImageData(
                    path=os.path.join(record.relative_path, record.image_file_name),
                    album_name=album,
                    file_name=record.image_file_name,
                    created=record.creation_date,
                )
                yield img_data, record


def validate_albums(library_type: Supported, library_dir: str) -> Dict[str, Dict[str, Any]] | None:
    """Checks for album information in the given library. If no albums are found or the library_type type is not
    supported then None is returned.

    Parameters
    ----------
    library_type : Supported
        The photo library flavor to ingest
    library_dir : str
        Absolute path to the photo library

    Returns
    -------
    Dict[str, Dict[str, Any]] | None
    """

    albums = None
    if library_type == Supported.DIGIKAM_PHOTO_LIBRARY:
        with DigikamReader(photolibrary_path=library_dir) as db:
            albums = db.albums
    elif library_type == Supported.MACOS_PHOTO_LIBRARY:
        with MacPhotosReader(photolibrary_path=library_dir) as db:
            albums = db.albums
    return albums


def build(
    library_type: Supported,
    library_dir: str,
    chroma_path: str,
    albums: List[str],
    geonames_user: str = os.getenv("GEONAMES_USERNAME"),
) -> int:
    """Database builder

    Parameters
    ----------
    library_type : Supported
        The photo library flavor to ingest
    library_dir : str
        Absolute path to the photo library
    chroma_path : str
        Absolute path to the directory in which to save the ChromaDB assets
    albums : List[str]
        Albums to process
    geonames_user : str
        Registered Geonames API username, by default os.getenv("GEONAMES_USERNAME")

    Returns
    -------
    int
        Number of records stored in the vector database

    Raises
    ------
    TypeError
        Raised if the photo library is unsupported
    """

    if library_type == Supported.DIGIKAM_PHOTO_LIBRARY:
        streamer = stream_digikam_albums
    elif library_type == Supported.MACOS_PHOTO_LIBRARY:
        streamer = stream_macos_albums
    else:
        raise TypeError(f"{library_type.value} is not yet supported")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    captioner = ImageCaption(device=device, batch_size=16)
    rev_geo_coder = GeonamesReverseGeocoder(geonames_user=geonames_user)

    vector_store = ImageVectorStore(chroma_persist_path=chroma_path, model_kwargs={"device": device})

    image_batch = []
    for image, metadata in streamer(photo_library_dir=library_dir, albums=albums):
        image = generate_geo_descriptions(image, metadata, geocoder=rev_geo_coder)
        image = generate_people_in_scene_descriptions(image, metadata)

        image_batch.append(image)

        if len(image_batch) > 256:
            image_batch = batch_caption(image_batch, captioner)
            vector_store.add_images(image_batch)
            image_batch.clear()

    if len(image_batch) > 0:
        image_batch = batch_caption(image_batch, captioner)
        vector_store.add_images(image_batch)
        image_batch.clear()

    rev_geo_coder.teardown()
    return len(vector_store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--geonames_user", type=str, help="Username for Geonames API")
    parser.add_argument("--type", type=Supported.argparse, choices=list(Supported))
    parser.add_argument("--photo_lib_path", type=str, help="Absolute path to the photo library to process")
    parser.add_argument("--chroma_path", type=str, help="Override the path to the ChromaDB database", required=False)
    parser.add_argument("--album", action="append", help="Album name to process")
    args = parser.parse_args()

    if not args.album:
        available_albums = validate_albums(args.type, args.photo_lib_path)
        if available_albums is None:
            raise TypeError(f"`{args.type}` is not supported")

        available = '\n ** '.join(k for k, v in available_albums.items() if v["count"] > 0)
        raise AttributeError(
            "No album(s) were provided. "
            f"Albums available: \n ** {available}"
        )

    build(
        library_type=args.type,
        library_dir=args.photo_lib_path,
        chroma_path=args.chroma_path,
        albums=args.album,
        geonames_user=args.geonames_user
    )
