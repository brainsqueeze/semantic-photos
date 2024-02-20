from typing import List, Tuple
import warnings
import argparse
import os

from PIL import Image
import pillow_heif

import gradio as gr

from semantic_photos.models.documents import ImageVectorStore

warnings.filterwarnings("ignore")
pillow_heif.register_heif_opener()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--chroma_path", type=str, help="Override the path to the ChromaDB database", required=False)
args = parser.parse_args()

chroma_path = args.chroma_path if args.chroma_path is not None else os.getenv("MODEL_CACHE_DIR")
photo_store = ImageVectorStore(chroma_persist_path=chroma_path)


def search(query: str) -> List[Tuple[str, str]]:
    """Search function

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
    scale = 0.1

    output = []
    for hit, score in hits:
        img = Image.open(hit.metadata["path"])
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        output.append((img, f"Score: {round(score, 2)}"))

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
                type="pil"
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
