from huggingface_models.Phi3_5_v.modeling_phi3_v import Phi3VForCausalLM
from huggingface_models.Phi3_5_v.processing_phi3_v import Phi3VProcessor

from huggingface_models.Phi3_5_v.skatr_vit_phi3_v import Phi3ImageEmbeddingSkatr
from custom.Phi3_v_skatr_processor.skatr_processor_phi3_v import Phi3VProcessorSkatr

from transformers import AutoProcessor

def convert_phi3_to_skatr(model, patch_size, backbone_dir=None):
    model.model.config.img_processor = {
        "image_dim_out": 160,
        "model_name": "vit_skatr",
        "name": "vit_skatr",
        "patch_size": patch_size
    }
    if not backbone_dir:
        backbone_dir = "/remote/gpu03/schiller/skatr/runs/regression_micro_ViT/2024-10-02_12-00-27"
    embedding_config = {
        'embedding_cls': model.config.embd_layer['embedding_cls'],
        **model.config.embd_layer
    }
    model.model.vision_embed_tokens = Phi3ImageEmbeddingSkatr(
        model.config,
        backbone_dir=backbone_dir,
        wte=model.model.embed_tokens,
        **embedding_config
    )

    processor = AutoProcessor.from_pretrained(
        "/remote/gpu03/schiller/skatr/models/custom/Phi3_v_skatr_processor", 
        trust_remote_code=True,
        local_files_only=True,
        patch_size = [4, 4, 10]
    )
    # lightcone_param_token_dict = {"additional_special_tokens": processor.tokenizer.additional_special_tokens + [processor.get_lightcone_param_token()]}
    # processor.tokenizer.add_special_tokens(lightcone_param_token_dict)
    # model.model.resize_token_embeddings(1)

    return model, processor
