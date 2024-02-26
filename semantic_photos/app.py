from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import cache
import warnings
import argparse
import os

from PIL import Image, ExifTags
import gradio as gr

from semantic_photos.models.documents import ImageVectorStore

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    warnings.warn("pillow-heif is not installed, HEIC images will not render")

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--chroma_path", type=str, help="Override the path to the ChromaDB database", required=False)
args = parser.parse_args()

chroma_path = args.chroma_path if args.chroma_path is not None else os.getenv("MODEL_CACHE_DIR")
photo_store = ImageVectorStore(chroma_persist_path=chroma_path)
OUTPUT_TYPE = "pil"


@cache
def _get_rotation_key() -> int:
    return max(ExifTags.TAGS.items(), key=lambda x: x[1] == 'Orientation', default=(-1, None))[0]


def search(query: str) -> List[Tuple[str, str]]:
    """Search function. If sending PIL Image objects then this function attempts to autocorrect the orientation. It also
    sends a downscaled version of the image.

    Parameters
    ----------
    query : str
        Search query

    Returns
    -------
    List[Tuple[str, str]]
        (absolute image path, score text)
    """

    hits = photo_store.query(query, n_results=12)

    def _load(hit):
        scale = 0.3
        score = None
        if isinstance(hit, (tuple, list)):
            hit, score = hit
        img = Image.open(hit.metadata["path"])

        if hasattr(img, '_getexif'):
            orientation_key = _get_rotation_key()
            e = getattr(img, '_getexif')()
            if e is not None:
                if e.get(orientation_key) == 3:
                    img = img.transpose(Image.ROTATE_180)  # pylint: disable=no-member
                elif e.get(orientation_key) == 6:
                    img = img.transpose(Image.ROTATE_270)  # pylint: disable=no-member
                elif e.get(orientation_key) == 8:
                    img = img.transpose(Image.ROTATE_90)  # pylint: disable=no-member
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))

        if isinstance(score, (float, int)):
            return img, f"Score: {round(score, 2)}"
        return img, None

    if OUTPUT_TYPE == "filepath":
        output = [(hit.metadata["path"], f"Score: {round(score, 2)}") for hit, score in hits]
    else:
        output = []
        with ThreadPoolExecutor() as executor:
            for o in executor.map(_load, hits):
                output.append(o)

    return output


def build_app() -> gr.Blocks:
    """Gradio app builder

    Returns
    -------
    gr.Blocks
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="Semantic photo search") as demo:
        gr.Markdown(
            "<h1>Semantic photos search</h1>"
            "Run a query to see relevant photos with the relevance score (lower scores are better)."
        )
        with gr.Column():
            query_bar = gr.Textbox(lines=1, label="Search")

            gallery = gr.Gallery(
                label="Photo hits",
                show_label=True,
                columns=[4],
                rows=[3],
                object_fit="contain",
                height="75vh",
                interactive=False,
                type=OUTPUT_TYPE
            )

        # pylint: disable=no-member
        query_bar.submit(
            fn=search,
            inputs=[query_bar],
            outputs=[gallery],
        )
    return demo


if __name__ == '__main__':
    app = build_app()
    app.queue(max_size=10).launch(server_name="0.0.0.0")
