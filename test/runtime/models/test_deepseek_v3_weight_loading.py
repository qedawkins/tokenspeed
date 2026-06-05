from tokenspeed.runtime.layers.quantization import Fp8Config, Mxfp4Config
from tokenspeed.runtime.models.deepseek_v3 import _qkv_a_quant_block_size


def test_qkv_a_quant_block_size_defaults_for_mxfp4_config() -> None:
    assert _qkv_a_quant_block_size(Mxfp4Config()) == 1


def test_qkv_a_quant_block_size_uses_fp8_weight_block_size() -> None:
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[16, 32],
    )

    assert _qkv_a_quant_block_size(quant_config) == 16
