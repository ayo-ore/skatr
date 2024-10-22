# Copied and modified from microsoft/Phi-3.5-vision-instruct (https://huggingface.co/microsoft/Phi-3.5-vision-instruct/resolve/main/config.json), configuration_phi3_v.py

""" LLM-ViT model configuration"""


from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

DEFAULT_IMAGE_PROC_CONFIG = {
    "image_dim_out": 160,
    "model_name": "vit",
    "name": "vit_skatr",
    "patch_size": [
      4,
      4,
      10
    ]
}

DEFAULT_EMBED_LAYER_CONFIG = {
    "embedding_cls": "image",
    "hd_transform_order": "sub_glb",
    "projection_cls": "mlp",
    "use_hd_transform": False,
    "with_learnable_separator": True
}

DEFAULT_ROPE_SCALING_CONFIG = {
    "long_factor": [
      1.0800000429153442,
      1.1100000143051147,
      1.1399999856948853,
      1.340000033378601,
      1.5899999141693115,
      1.600000023841858,
      1.6200000047683716,
      2.620000123977661,
      3.2300000190734863,
      3.2300000190734863,
      4.789999961853027,
      7.400000095367432,
      7.700000286102295,
      9.09000015258789,
      12.199999809265137,
      17.670000076293945,
      24.46000099182129,
      28.57000160217285,
      30.420001983642578,
      30.840002059936523,
      32.590003967285156,
      32.93000411987305,
      42.320003509521484,
      44.96000289916992,
      50.340003967285156,
      50.45000457763672,
      57.55000305175781,
      57.93000411987305,
      58.21000289916992,
      60.1400032043457,
      62.61000442504883,
      62.62000274658203,
      62.71000289916992,
      63.1400032043457,
      63.1400032043457,
      63.77000427246094,
      63.93000411987305,
      63.96000289916992,
      63.970001220703125,
      64.02999877929688,
      64.06999969482422,
      64.08000183105469,
      64.12000274658203,
      64.41000366210938,
      64.4800033569336,
      64.51000213623047,
      64.52999877929688,
      64.83999633789062
    ],
    "short_factor": [
       1.08,
      1.1,
      1.1300000000000001,
      1.2800000000000002,
      1.3100000000000003,
      1.4500000000000004,
      1.4500000000000004,
      1.9500000000000008,
      2.030000000000001,
      2.4299999999999926,
      2.5699999999999896,
      2.9499999999999815,
      3.729999999999965,
      3.869999999999962,
      4.189999999999955,
      4.43999999999995,
      4.6399999999999455,
      4.979999999999938,
      5.159999999999934,
      5.279999999999932,
      5.759999999999922,
      5.889999999999919,
      5.889999999999919,
      5.969999999999917,
      6.089999999999915,
      6.2799999999999105,
      6.7699999999999,
      6.8899999999998975,
      7.109999999999893,
      7.129999999999892,
      7.179999999999891,
      7.289999999999889,
      7.339999999999888,
      7.559999999999883,
      7.619999999999882,
      7.69999999999988,
      7.879999999999876,
      7.879999999999876,
      7.879999999999876,
      7.939999999999875,
      7.949999999999875,
      7.979999999999874,
      8.19999999999987,
      8.439999999999864,
      8.469999999999864,
      8.589999999999861,
      8.809999999999857,
      8.999999999999853
    ],
    "type": "su"
}

class LLMViTConfig(PretrainedConfig):
   
    model_type = "llm_vit"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=32000,
        sliding_window=262144,
        embd_layer: dict = DEFAULT_EMBED_LAYER_CONFIG,
        img_processor: dict = DEFAULT_IMAGE_PROC_CONFIG,
        rope_scaling: dict = DEFAULT_ROPE_SCALING_CONFIG,
        _attn_implementation = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.sliding_window = sliding_window
        self.embd_layer = embd_layer
        self.img_processor = img_processor
        self.rope_scaling = rope_scaling
        self._attn_implementation = _attn_implementation

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 3:
            raise ValueError(
                "`rope_scaling` must be a dictionary with three fields, `type`, `short_factor` and `long_factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_short_factor = self.rope_scaling.get("short_factor", None)
        rope_scaling_long_factor = self.rope_scaling.get("long_factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["su", "yarn"]:
            raise ValueError(f"`rope_scaling`'s type field must be one of ['su', 'yarn'], got {rope_scaling_type}")
        if not (
            isinstance(rope_scaling_short_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_short_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s short_factor field must be a list of numbers, got {rope_scaling_short_factor}"
            )
        if not len(rope_scaling_short_factor) == self.hidden_size // self.num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s short_factor field must have length {self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_short_factor)}"
            )
        if not (
            isinstance(rope_scaling_long_factor, list)
            and all(isinstance(x, (int, float)) for x in rope_scaling_long_factor)
        ):
            raise ValueError(
                f"`rope_scaling`'s long_factor field must be a list of numbers, got {rope_scaling_long_factor}"
            )
        if not len(rope_scaling_long_factor) == self.hidden_size // self.num_attention_heads // 2:
            raise ValueError(
                f"`rope_scaling`'s long_factor field must have length {self.hidden_size // self.num_attention_heads // 2}, got {len(rope_scaling_long_factor)}"
            )
