"""CHAIR evaluation for LLaVA-v1.5-7B.

Supports: vanilla, ggd (ours), ONLY, ONLY+EIC, VCD, M3ID, ASCD.
"""
import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))
_self_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _self_dir]

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM

from ggd.eval_common import (
    caption_output_path,
    chair_protocol_image_files,
    excluded_image_ids,
    load_eic_scores,
    resolve_method,
    select_image_files,
    validate_method_flags,
)
from ggd.models.llava_sampling import (
    evolve_only_sampling,
    install_ascd_llava15,
)
from ggd.monitor import CausalLogitsProcessor, CausalMonitor
from ggd.only_eic import inject_eic_for_only
from ggd.vcd import add_diffusion_noise

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="CHAIR eval for LLaVA-v1.5-7B")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--image_seed", type=int)
    p.add_argument("--per_image_seed", action="store_true")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--anno_path", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True,
                   help="Output directory or .jsonl file path")
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--exclude_image_ids_file", type=str)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument(
        "--attn_implementation",
        choices=["auto", "eager", "sdpa"],
        default="auto",
    )
    p.add_argument(
        "--record_efficiency",
        action="store_true",
        help="Record synchronized generation latency and peak CUDA memory per image",
    )
    p.add_argument("--conv_mode", type=str, default="v1")
    p.add_argument(
        "--caption_prompt",
        default="Please describe this image in detail.",
    )

    p.add_argument("--eic_scores_path", type=str, default=None)
    p.add_argument("--layer_index", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--img_start", type=int, default=35)
    p.add_argument("--img_len", type=int, default=576)
    p.add_argument("--method_name", type=str, default=None,
                   help="Override output method tag (default: inferred from flags)")

    p.add_argument("--no_hook", action="store_true",
                   help="Disable GGD monitor (vanilla or ONLY/VCD/M3ID)")
    p.add_argument("--use_only", action="store_true", help="ONLY baseline")
    p.add_argument("--use_eic_heads", action="store_true",
                   help="With --use_only: use offline EIC head set in CD branch")
    p.add_argument("--use_vcd", action="store_true", help="VCD baseline")
    p.add_argument("--use_m3id", action="store_true", help="M3ID baseline")
    p.add_argument(
        "--use_ascd",
        action="store_true",
        help="ASCD baseline (official LLaVA-1.5-7B settings)",
    )
    p.add_argument("--ascd_alpha", type=float, default=1.0)
    p.add_argument("--ascd_beta", type=float, default=0.1)
    p.add_argument("--noise_step", type=int, default=500)

    p.add_argument("--js_gamma", type=float, default=0.2)
    p.add_argument("--ritual_alpha_pos", type=float, default=3.0)
    p.add_argument("--ritual_alpha_neg", type=float, default=1.0)
    p.add_argument("--ritual_beta", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    validate_method_flags(args)
    method, needs_scores = resolve_method(args)
    # `method` is the canonical decoding behavior (vanilla/only/only_eic/vcd/m3id/
    # ascd/ggd) derived from flags. `--method_name` is only an OUTPUT LABEL (e.g.
    # "ggd_random" for head-selection ablations) and must NOT change behavior.
    label = args.method_name if args.method_name else method
    if needs_scores and not args.eic_scores_path:
        raise ValueError(f"--eic_scores_path is required for method={method}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    # GGD needs returned attention weights; ASCD modifies pre-softmax attention
    # logits. Both therefore require the eager implementation.
    attn_impl = (
        "eager"
        if method in ("ggd", "ascd")
        else "sdpa"
    ) if args.attn_implementation == "auto" else args.attn_implementation
    if method == "ascd" and attn_impl != "eager":
        raise ValueError("ASCD requires --attn_implementation eager")
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    evolve_only_sampling()
    if method == "ascd":
        installed = install_ascd_llava15(
            model,
            image_start=args.img_start,
            image_length=args.img_len,
        )
        log.info(
            "ASCD installed on %d layers: upstream=70034c32 alpha=%s beta=%s",
            installed,
            args.ascd_alpha,
            args.ascd_beta,
        )

    monitor = None
    orig_fwd = None
    processors = LogitsProcessorList([])
    layer_for_only = args.layer_index

    if needs_scores:
        if not args.eic_scores_path:
            raise ValueError(f"--eic_scores_path is required for method={method}")
        eic_scores = load_eic_scores(args.eic_scores_path, args.layer_index)
        log.info(
            f"EIC scores layer={args.layer_index}: "
            f"nonzero={int((eic_scores > 0).sum())}/{len(eic_scores)}"
        )

        if method == "ggd":
            monitor = CausalMonitor(
                model, args.layer_index, eic_scores,
                img_start=args.img_start,
                img_len=args.img_len,
            )
            orig_fwd = monitor.install_qk_hook()
            processors = LogitsProcessorList(
                [CausalLogitsProcessor(monitor, alpha=args.alpha)]
            )
        elif method == "only_eic":
            layer_for_only = inject_eic_for_only(
                model=model,
                scores_path=args.eic_scores_path,
                layer_index=args.layer_index,
                pure_eic=True,
            )

    with open(args.anno_path) as f:
        coco = json.load(f)
    image_seed = args.seed if args.image_seed is None else args.image_seed
    exclusions = excluded_image_ids(args.exclude_image_ids_file)
    selected_files = select_image_files(
        chair_protocol_image_files(
            coco["images"],
            "llava",
            image_seed=args.image_seed,
            exclusions_file=args.exclude_image_ids_file,
        ),
        exclusions,
        args.num_eval_samples,
        image_seed,
    )
    images_by_filename = {
        image["file_name"]: image for image in coco["images"]
    }
    images = [images_by_filename[filename] for filename in selected_files]

    out_file = caption_output_path(args.out_path, label)
    log.info(f"CHAIR method={method} label={label} alpha={args.alpha} n={len(images)} -> {out_file}")

    results = []
    for img_info in tqdm(images, total=len(images)):
        img_id = img_info["id"]
        img_path = os.path.join(args.data_path, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).half().to(model.device)
        if monitor is not None:
            monitor.reset()

        qs = DEFAULT_IMAGE_TOKEN + "\n" + args.caption_prompt
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
        ).unsqueeze(0).to(model.device)

        image_neg = None
        if method == "vcd":
            image_neg = add_diffusion_noise(image_tensor, args.noise_step)

        gen_kwargs = dict(
            images=image_tensor,
            images_neg=image_neg,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            use_only=(method in ("only", "only_eic")),
            use_vcd=(method == "vcd"),
            use_m3id=(method == "m3id"),
            use_ascd=(method == "ascd"),
            ascd_alpha=args.ascd_alpha,
            ascd_beta=args.ascd_beta,
            enhance_layer_index=layer_for_only,
            js_gamma=args.js_gamma,
            ritual_alpha_pos=args.ritual_alpha_pos,
            ritual_alpha_neg=args.ritual_alpha_neg,
            ritual_beta=args.ritual_beta,
        )
        if processors:
            gen_kwargs["logits_processor"] = processors

        if args.per_image_seed:
            sample_seed = (args.seed * 1_000_003 + img_id) % (2**31)
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            random.seed(sample_seed)
            np.random.seed(sample_seed)

        if args.record_efficiency and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(input_ids, **gen_kwargs)
        if args.record_efficiency and torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_start
        peak_memory_gib = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if args.record_efficiency and torch.cuda.is_available()
            else None
        )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]

        generated_tokens = int(output_ids.shape[1] - input_ids.shape[1])
        output_text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True,
        )[0].strip()
        result = {
            "image_id": img_id,
            "caption": output_text,
            "caption_prompt": args.caption_prompt,
        }
        if args.record_efficiency:
            result["generation_seconds"] = generation_seconds
            result["peak_memory_gib"] = peak_memory_gib
            result["generated_tokens"] = generated_tokens
        results.append(result)

    if monitor is not None and orig_fwd is not None:
        monitor.restore(orig_fwd)

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log.info(f"[done] Saved {len(results)} captions to {out_file}")

if __name__ == "__main__":
    main()
