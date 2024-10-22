import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-bb", "--vit_backbone_dir", help="Backbone dir for the skatr vit")
parser.add_argument("-sd", "--save_dir", help="Directory to save the model")
args = parser.parse_args()

from transformers import AutoProcessor
from src.networks.llm_vit_construct.modeling_llm_vit import LLMViTForCausalLM
from src.networks.llm_vit_construct.configuration_llm_vit import LLMViTConfig
import shutil

import os, json
from transformers.utils.__init__ import SAFE_WEIGHTS_INDEX_NAME
from transformers.modeling_utils import load_state_dict, set_module_tensor_to_device
from accelerate.utils.modeling import find_tied_parameters
# from safetensors.torch import load_file
import gc
import fnmatch
import torch

def run():
    load_dir = "/remote/gpu03/schiller/skatr/src/networks/llm_vit_construct"

    model = LLMViTForCausalLM(LLMViTConfig(device_map="cpu"))
    model.load_vit_weights(args.bb)

    data_dir = "/remote/gpu03/schiller/skatr/models/huggingface_models/Phi3_5_v"

    archive_file = os.path.join(data_dir, SAFE_WEIGHTS_INDEX_NAME)
    modules_to_load = [
    "lm_head.weight",
    "model.layers.*",
    "model.norm.weight"
    ]

    def _to_be_loaded(name:str, modules_to_load:list[str]) -> False:
        for module in modules_to_load:
            if fnmatch.fnmatch(name, module):
                return True
        return False

    with open(archive_file, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    shard_filenames = [os.path.join(data_dir, f) for f in shard_filenames]

    model.tie_weights()

    tied_params = find_tied_parameters(model)

    for shard_file in shard_filenames:
        state_dict = load_state_dict(shard_file, False)

        state_dict_to_load = {k: v for k,v in state_dict.items() if _to_be_loaded(k, modules_to_load)}
        model.load_state_dict(state_dict_to_load, strict=False)

        del state_dict
        gc.collect()

    processor = AutoProcessor.from_pretrained(load_dir, trust_remote_code=True, local_files_only=True)

    save_dir = args.save_dir
    shutil.rmtree(save_dir)

    model.save_pretrained(save_directory=save_dir)
    processor.save_pretrained(save_directory=save_dir)


if __name__ == '__main__':
    run()