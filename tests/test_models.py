import os

import torch
from test_smoke import tiny_model

from semi_dpo.models import load_unet


def perturbed_unet(base):
    unet = load_unet(base)
    with torch.no_grad():
        for param in unet.parameters():
            param.add_(0.5)
    return unet


def same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_load_unet_checkpoint_formats(tmp_path):
    base = tiny_model("sd15")
    trained = perturbed_unet(base)

    # FSDP full state dict: checkpoint-N/unet/unet.pt
    fsdp_dir = tmp_path / "fsdp" / "checkpoint-1"
    os.makedirs(fsdp_dir / "unet")
    torch.save(trained.state_dict(), fsdp_dir / "unet" / "unet.pt")

    # diffusers format: checkpoint-N/unet/{config.json, weights}
    diffusers_dir = tmp_path / "ddp" / "checkpoint-1"
    trained.save_pretrained(diffusers_dir / "unet")

    for path in (fsdp_dir, diffusers_dir, diffusers_dir / "unet"):
        assert same_weights(load_unet(base, str(path)), trained), path
    assert not same_weights(load_unet(base), trained)


def test_load_unet_lora_checkpoint(tmp_path):
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from semi_dpo.models import LORA_WEIGHT_NAME

    base = tiny_model("sd15")
    unet = load_unet(base)
    unet.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=["to_q", "to_v"]))
    with torch.no_grad():
        for name, param in unet.named_parameters():
            if "lora_B" in name:
                param.normal_()
    StableDiffusionPipeline.save_lora_weights(
        tmp_path, unet_lora_layers=get_peft_model_state_dict(unet), weight_name=LORA_WEIGHT_NAME
    )

    merged = load_unet(base, str(tmp_path))
    unet.fuse_lora()
    unet.unload_lora()
    assert same_weights(merged, unet)
    assert not same_weights(merged, load_unet(base))
