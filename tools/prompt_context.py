"""Precompute the prompt context that the LTX-2.5 transformer consumes.

The Rust pipeline does not run the Gemma text encoder. This script runs the
reference ``PromptEncoder`` once (Gemma, then the embeddings processor whose
connector weights live in the transformer checkpoint) and saves its output as a
safetensors file that ``ltx`` loads with ``--prompt-context``.

Run with the reference venv:

    python -W ignore tools/prompt_context.py \
        --transformer ltx-2.5-transformer.safetensors \
        --text-encoder /path/to/gemma \
        --prompt "..." \
        --out prompt_context.safetensors

File layout (all tensors batch 1):

    positive.video_encoding   [1, S, D]  bf16   EmbeddingsProcessorOutput.video_encoding
    positive.attention_mask   [1, S]     f32    1.0 = valid token, 0.0 = padding
    negative.video_encoding   [1, S, D]  bf16
    negative.attention_mask   [1, S]     f32

Metadata: ``format`` = "ltx-prompt-context/1", ``prompt``, ``negative_prompt``,
``transformer`` and ``text_encoder`` (file names), ``ltx2_commit``.

Reference: ``ltx_pipelines.utils.blocks.PromptEncoder`` and
``ltx_pipelines.alpha_gen.AlphaGenPipeline.__call__`` (``ctx_p, ctx_n``) at LTX-2
commit 9ec55f9.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

LTX2_COMMIT = "9ec55f9"
FORMAT = "ltx-prompt-context/1"


def main() -> None:
    from ltx_pipelines.alpha_gen import NEGATIVE_PROMPT
    from ltx_pipelines.utils.blocks import PromptEncoder
    from ltx_pipelines.utils.helpers import get_device
    from ltx_pipelines.utils.model_paths import ModelPaths

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--transformer", required=True, help="LTX-2.5 transformer safetensors (holds the connector).")
    parser.add_argument("--text-encoder", required=True, help="Gemma text encoder root or safetensors.")
    parser.add_argument("--prompt", required=True, help="Positive prompt.")
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT, help="Negative prompt (alpha_gen default).")
    parser.add_argument("--out", required=True, type=Path, help="Output safetensors path.")
    args = parser.parse_args()

    model_paths = ModelPaths.from_split(
        transformer_path=args.transformer,
        text_encoder_path=args.text_encoder,
    )
    # alpha_gen encodes prompts in bf16 on the pipeline device.
    encoder = PromptEncoder(model_paths, dtype=torch.bfloat16, device=get_device())
    with torch.inference_mode():
        positive, negative = encoder([args.prompt, args.negative_prompt])

    tensors = {}
    for name, output in (("positive", positive), ("negative", negative)):
        tensors[f"{name}.video_encoding"] = output.video_encoding.detach().to("cpu", torch.bfloat16).contiguous()
        tensors[f"{name}.attention_mask"] = output.attention_mask.detach().to("cpu", torch.float32).contiguous()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.out),
        metadata={
            "format": FORMAT,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "transformer": Path(args.transformer).name,
            "text_encoder": Path(args.text_encoder).name,
            "ltx2_commit": LTX2_COMMIT,
        },
    )
    for key, value in tensors.items():
        print(f"{key}: {tuple(value.shape)} {value.dtype}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
