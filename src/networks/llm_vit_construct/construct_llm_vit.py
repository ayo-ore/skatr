import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-bb", "--vit_backbone_dir", help="Backbone dir for the skatr vit")
parser.add_argument("-sd", "--save_dir", help="Directory to save the model")
args = parser.parse_args()

from transformers import AutoModelForCausalLM, AutoProcessor
import shutil

def run():
    load_dir = "/remote/gpu03/schiller/skatr/src/networks/llm_vit_construct"

    model = AutoModelForCausalLM.from_pretrained(load_dir, trust_remote_code=True, local_files_only=True)
    model.load_vit_weights(args.bb)
    processor = AutoProcessor.from_pretrained(load_dir, trust_remote_code=True, local_files_only=True)

    save_dir = args.save_dir
    shutil.rmtree(save_dir)

    model.save_pretrained(save_directory=save_dir)
    processor.save_pretrained(save_directory=save_dir)


if __name__ == '__main__':
    run()