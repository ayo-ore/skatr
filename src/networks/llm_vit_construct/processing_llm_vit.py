# Copied and modified from microsoft/Phi-3.5-vision-instruct (https://huggingface.co/microsoft/Phi-3.5-vision-instruct/resolve/main/config.json), processing_phi3_v.py

"""
Processor class for LLM-ViT.
"""
import re
from typing import List, Optional, Union

import torch

import transformers
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType


"""Image processor class for LLM ViT."""

from typing import List, Optional, Union

import numpy as np

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    make_list_of_images,
    valid_images,
    is_numpy_array,
    is_torch_tensor
)
from transformers.utils import TensorType, is_vision_available, logging

from transformers import AutoImageProcessor

logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image

import torch
import torchvision

class LLMViTImageProcessor(BaseImageProcessor):
    
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        patch_size=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size
        
    def calc_num_image_tokens(
            self, 
            images: ImageInput 
    ):
        images = make_list_of_images(images)

        if is_numpy_array(images[0]):
            images = [torch.from_numpy(img) for img in images]
        elif not is_torch_tensor(images[0]):
            raise ValueError(
                "Invalid image type. Must be of type numpy.ndarray or torch.Tensor."
            )
        
        assert len(images[0].dim()) == len(self.patch_size)

        shapes = [[im.size()[i] for i in range(im.dim())] for im in images]
        num_img_tokens = [ np.prod([ shape_size // dim_patch_size for dim_patch_size, shape_size in zip(self.patch_size, shape)]) for shape in shapes]
        return num_img_tokens

    def calc_num_image_tokens_from_image_size(self, shape):
        
        assert len(self.patch_size) == len(shape)

        num_img_tokens = np.prod([ shape_size // dim_patch_size for dim_patch_size, shape_size in zip(self.patch_size, shape)]) 
        return num_img_tokens

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):

        images = make_list_of_images(images)

        if is_numpy_array(images[0]):
            images = [torch.from_numpy(im).unsqueeze(-4) for im in images]
        elif is_torch_tensor(images[0]):
            images = [im.unsqueeze(-4) for im in images]
        else:
            raise ValueError(
                "Invalid image type. Must be of type numpy.ndarray or torch.Tensor."
            )

        shapes = [[im.size()[i] for i in range(1, im.dim())] for im in images] # First dimension is the number of channels = 1
        num_img_tokens = [ int(np.prod([ shape_size // dim_patch_size for dim_patch_size, shape_size in zip(self.patch_size, shape)])) for shape in shapes]

        images = torch.stack(images, dim=0)

        data = {"pixel_values": images, 
                "image_sizes": shapes,
                "num_img_tokens": num_img_tokens
                }

        return BatchFeature(data=data, tensor_type=return_tensors)

AutoImageProcessor.register("LLMViTImageProcessor", LLMViTImageProcessor)

transformers.LLMViTImageProcessor = LLMViTImageProcessor 

class LLMViTProcessor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LLMViTImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    special_image_token = "<|lightcone|>"
    lightcone_param_token = "<|placeholder1|>"

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.img_tokens = [f"<|lightcone_{i+1}|>" for i in range(1000000)]

    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if images is not None:
            image_inputs = self.image_processor(images, return_tensors=return_tensors)
        else:
            image_inputs = {}
        inputs = self._convert_images_texts_to_inputs(image_inputs, text, padding=padding, truncation=truncation, max_length=max_length, return_tensors=return_tensors)
        return inputs

    def calc_num_image_tokens(self, images: ImageInput):
        return self.image_processor.calc_num_image_tokens(images)
        
    def calc_num_image_tokens_from_image_size(self, shape):
        return self.image_processor.calc_num_image_tokens_from_image_size(shape)
    
    
    @property 
    def special_image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.special_image_token)

    def get_lightcone_param_token(self):
        return self.lightcone_param_token

    def get_special_image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.special_image_token)
    
    def lightcone_param_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.lightcone_param_token)

    def _convert_images_texts_to_inputs(self, images, texts, padding=False, truncation=None, max_length=None, return_tensors=None):

        if not len(images):
            model_inputs = self.tokenizer(texts, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length)
            return BatchFeature(data={**model_inputs})

        pattern = r"<\|lightcone_\d+\|>"
        prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in re.split(pattern, texts)] 

        if 'num_img_tokens' in images:
            num_img_tokens = images['num_img_tokens']
        else:
            num_img_tokens = self.calc_num_image_tokens(images)

        images, image_sizes = images['pixel_values'], images['image_sizes']

        # image_tags needs to start from 1 to n
        image_tags = re.findall(pattern, texts) 
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        assert len(unique_image_ids) == len(images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(images)} images"

        image_ids_pad = [[-iid]*num_img_tokens[iid-1] for iid in image_ids]
        
        def insert_separator(X, sep_list):
            if len(X) > len(sep_list):
                sep_list.append([])
            return [ele for sublist in zip(X, sep_list) for ele in sublist]
        input_ids = []
        offset = 0
        for x in insert_separator(prompt_chunks, image_ids_pad):
            input_ids.extend(x[offset:])

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = (input_ids > -1000000).to(torch.long)

        return BatchFeature(data={"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "pixel_values": images, 
                                  "image_sizes": image_sizes})


    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
