"""
CHAIR evaluation for InternVL3.5-8B-HF.
"""
import os, sys, json, argparse, logging, warnings, random, time
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transformers" / "src"))
sys.path.insert(0, str(REPO))

from ggd.transformers_fork import ensure_internvl_fork
ensure_internvl_fork()

import torch
import numpy as np
from PIL import Image

from transformers import AutoProcessor
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.internvl.modeling_internvl_real import (
    InternVLForConditionalGeneration,
)

from ggd.eval_common import (
    chair_protocol_image_files,
    excluded_image_ids,
    select_image_files,
)
from ggd.models.internvl import evolve_only_sampling_internvl
from ggd.monitor import (
    CausalLogitsProcessor,
    CausalMonitorInternVL,
    parse_image_id,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    p = argparse.ArgumentParser(description="CHAIR eval for InternVL3.5-8B-HF")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--image_seed", type=int)
    p.add_argument("--per_image_seed", action="store_true")
    p.add_argument("--model_path", type=str,
                   default=str(REPO / "data/models/InternVL3_5-8B-HF"))
    p.add_argument("--data_path", type=str,
                   default=str(REPO / "data/coco/val2014"))
    p.add_argument("--anno_path", type=str,
                   default=str(REPO / "data/coco/annotations/instances_val2014.json"))
    p.add_argument("--exclude_image_ids_file", type=str)
    p.add_argument("--log_path", type=str, default=None)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--eic_scores_path", type=str, required=True)
    p.add_argument("--layer_index", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Max temperature reduction when ungrounded")
    p.add_argument("--do_sample", type=str2bool, default=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--method_name", type=str, default="ggd")
    p.add_argument("--record_efficiency", action="store_true")
    p.add_argument("--no_hook", action="store_true",
                   help="Disable causal hook (vanilla run through same script)")
    p.add_argument("--use_only", action="store_true",
                   help="ONLY baseline via the patched _sample loop")
    p.add_argument("--use_vcd", action="store_true")
    p.add_argument("--use_m3id", action="store_true")
    p.add_argument("--noise_step", type=int, default=500)
    p.add_argument("--cd_alpha", type=float, default=1.0)
    p.add_argument("--cd_beta", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_path, exist_ok=True)
    if args.log_path:
        os.makedirs(args.log_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(args.log_path, "log.txt"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
    logger = log

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    model = InternVLForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto", trust_remote_code=False,
    )
    model.eval()
    image_token_id = model.config.image_token_id

    # The InternVL grounding monitor reads attention from the KV cache via a
    # manual Q*K (see CausalMonitorInternVL.install_hook), so it does NOT need
    # eager attention. Forcing eager here previously caused O(S^2) memory
    # blow-ups (CUDA OOM) on InternVL's high-resolution dynamic tiling, so we
    # keep the default memory-efficient sdpa kernel for all methods.

    evolve_only_sampling_internvl()

    payload = torch.load(args.eic_scores_path, map_location="cpu")
    if isinstance(payload, dict):
        eic_scores = payload.get("scores", payload.get("C", None))
        if eic_scores is None:
            eic_scores = next(iter(payload.values()))
    else:
        eic_scores = payload
    if eic_scores.dim() == 2:
        eic_scores = eic_scores[args.layer_index]
    eic_scores = eic_scores.float()
    log.info(f"EIC scores: {eic_scores.shape}, nonzero={int((eic_scores>0).sum())}/{len(eic_scores)}")

    monitor = None
    if args.no_hook or args.use_only:
        log.info("NO HOOK mode -- vanilla / ONLY run through same script")
        processors = LogitsProcessorList([])
    elif getattr(args, 'use_vcd', False) or getattr(args, 'use_m3id', False):
        processors = LogitsProcessorList([])
    else:
        monitor = CausalMonitorInternVL(model, args.layer_index, eic_scores, image_token_id)
        monitor.install_hook()
        causal_processor = CausalLogitsProcessor(monitor, alpha=args.alpha)
        processors = LogitsProcessorList([causal_processor])

    with open(args.anno_path) as handle:
        coco = json.load(handle)
    image_seed = args.seed if args.image_seed is None else args.image_seed
    image_files = chair_protocol_image_files(
        coco["images"],
        "internvl",
        image_seed=args.image_seed,
        exclusions_file=args.exclude_image_ids_file,
    )
    eval_files = select_image_files(
        image_files,
        excluded_image_ids(args.exclude_image_ids_file),
        args.num_eval_samples,
        image_seed,
    )
    logger.info(f"Evaluating {len(eval_files)} images from {args.data_path}")

    output_jsonl = os.path.join(
        args.out_path,
        f"ggd_alpha{args.alpha}_{args.method_name}.jsonl",
    )
    output_time = os.path.join(
        args.out_path,
        f"ggd_alpha{args.alpha}_{args.method_name}_time.txt",
    )
    open(output_jsonl, "w").close()
    open(output_time, "w").close()

    prompt = "Please describe this image in detail."

    for idx, img_file in tqdm(enumerate(eval_files), total=len(eval_files)):
        image_path = os.path.join(args.data_path, img_file)
        image_id = parse_image_id(img_file)

        raw_image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        if monitor is not None:
            monitor.set_img_positions(inputs.input_ids, image_token_id)

        if args.per_image_seed:
            sample_seed = (args.seed * 1_000_003 + image_id) % (2**31)
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
        if args.record_efficiency and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        if getattr(args, 'use_vcd', False) or getattr(args, 'use_m3id', False):
            from ggd.eval_common import import_vcd_baseline
            contrastive_generate, add_diffusion_noise = import_vcd_baseline("internvl")
            neg_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if args.use_vcd:
                neg_inputs["pixel_values"] = add_diffusion_noise(inputs["pixel_values"], args.noise_step)
            else:
                neg_inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
            generated_ids = contrastive_generate(
                model, dict(inputs), neg_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
                cd_alpha=args.cd_alpha, cd_beta=args.cd_beta,
            )
        else:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    use_only=bool(args.use_only),
                    enhance_layer_index=args.layer_index,
                    ritual_alpha_pos=3.0,
                    ritual_alpha_neg=1.0,
                    ritual_beta=0.1,
                    js_gamma=0.1,
                    logits_processor=processors,
                )
        if args.record_efficiency and torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        logger.info(f"[Causal-CHAIR InternVL]")
        logger.info(f"V: {image_path}")
        logger.info(f"Q: {prompt}")
        logger.info(f"A: {caption}")
        if monitor is not None:
            logger.info(f"grounding={monitor.grounding_score:.3f} entropy={monitor.mean_entropy:.3f}")
        logger.info("=" * 50)

        rec = {"image_id": int(image_id), "caption": caption}
        if args.record_efficiency:
            rec["generation_seconds"] = t2 - t1
            rec["peak_memory_gib"] = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else None
            )
            rec["generated_tokens"] = int(generated_ids_trimmed[0].numel())
        with open(output_jsonl, "a") as f:
            json.dump(rec, f)
            f.write("\n")
        with open(output_time, "a") as f:
            f.write(f"{t2 - t1}\n")

    if monitor is not None:
        monitor.restore()
    logger.info(f"Wrote captions: {output_jsonl}")
    logger.info(vars(args))

if __name__ == "__main__":
    main()
