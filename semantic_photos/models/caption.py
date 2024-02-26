from typing import List, Dict, Union, Optional, Any

from transformers import pipeline


class ImageCaption:
    """Image-to-text captioning class

    Parameters
    ----------
    model : str, optional
        Huggingface model name, by default "Salesforce/blip-image-captioning-base"
    device : str, optional
        Device onto which the model should be mapped, by default "cpu"
    model_kwargs : Optional[Dict[str, Any]], optional
        Optional pipeline kwargs, by default None
    batch_size : int, optional
        Max size of batch for multiple records, this is ignored for now, by default 8
    """

    def __init__(
        self,
        model: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu",
        model_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 8
    ):
        self.pipeline = pipeline(
            task="image-to-text",
            model=model,
            device=device,
            batch_size=batch_size,
            model_kwargs=model_kwargs
        )
        self.batch_size = batch_size

        self.__config = {
            "model_name": model,
            "batch_size": batch_size,
            "model_kwargs": model_kwargs,
            "task": "image-to-text"
        }

    def caption(self, images: Union[str, List[str]], **pipeline_kwargs) -> List[Dict[str, Any]]:
        """Runs the image captioning pipeline.

        Parameters
        ----------
        images : Union[str, List[str]]
            File path or list of image filepaths

        Returns
        -------
        List[Dict[str, Any]]
            image-to-text transformer output

        Raises
        ------
        TypeError
            Raised if the supplied image(s) is not a string or list of strings
        """

        pipeline_kwargs.setdefault("max_new_tokens", 32)

        output = []
        if isinstance(images, str):
            output.append(self.pipeline(images, **pipeline_kwargs)[0]["generated_text"])
        elif isinstance(images, (list, tuple)) and all(isinstance(img, str) for img in images):
            for out in self.pipeline(images, batch_size=self.batch_size, **pipeline_kwargs):
                output.append(' '.join(token["generated_text"] for token in out if token["generated_text"]))
        else:
            raise TypeError("`images` must be a string or list of strings")

        return output

    @property
    def config(self) -> Dict[str, Any]:
        """Parameters used for image-to-text inferencing.

        Returns
        -------
        Dict[str, Any]
        """

        return self.__config
