"""Patched LLaVA sampling for ONLY, VCD, M3ID, RITUAL, and ASCD.

Adapted from Wan et al.'s ONLY codebase (https://github.com/zifuwan/ONLY).
VCD: Leng et al., CVPR 2024 (https://github.com/DAMO-NLP-SG/VCD).
M3ID: Favero et al., CVPR 2024 (https://arxiv.org/abs/2403.14003).
ASCD: Wang et al. (https://github.com/BroJunn/ASCD), Apache-2.0,
upstream commit 70034c32edf2bcadb8aae57da2c4eaed82fab3e6.
"""
import copy
import warnings
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers

ASCD_HEADS = (
    (5, 15), (31, 27), (0, 5), (10, 17), (27, 15), (15, 11),
    (31, 4), (12, 31), (13, 8), (15, 25), (19, 16), (0, 26),
    (22, 26), (24, 6), (17, 15), (31, 0), (9, 27), (16, 15),
    (8, 1), (31, 31), (23, 1), (27, 12), (29, 29), (26, 13),
    (15, 4), (27, 17), (31, 13), (23, 11), (26, 9), (30, 17),
    (12, 6), (24, 27),
)


def install_ascd_llava15(model, image_start=35, image_length=576):
    base = model.get_model() if hasattr(model, "get_model") else model
    layers = getattr(base, "layers", None)
    if layers is None and hasattr(base, "model"):
        layers = getattr(base.model, "layers", None)
    if layers is None or len(layers) != 32:
        raise ValueError("ASCD requires the 32-layer LLaVA-1.5-7B model")

    heads_by_layer = {}
    for layer_idx, head_idx in ASCD_HEADS:
        heads_by_layer.setdefault(layer_idx, []).append(head_idx)

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        if int(attn.config.num_attention_heads) != 32:
            raise ValueError("ASCD requires 32 attention heads")
        mask = torch.zeros(32, dtype=torch.bool)
        mask[heads_by_layer.get(layer_idx, [])] = True
        attn._ascd_enabled = True
        attn._ascd_branch_id = 0
        attn._ascd_head_mask = mask
        attn._ascd_image_start = int(image_start)
        attn._ascd_image_length = int(image_length)
    return len(layers)


def set_ascd_branch(model, branch_id):
    if branch_id not in (0, 1):
        raise ValueError(f"Invalid ASCD branch: {branch_id}")
    modules = [
        module
        for module in model.modules()
        if getattr(module, "_ascd_enabled", False)
    ]
    if not modules:
        raise RuntimeError("ASCD is not installed")
    for module in modules:
        module._ascd_branch_id = branch_id


def ascd_contrastive_logits(logits, contrastive_logits, alpha=1.0, beta=0.1):
    if beta <= 0:
        raise ValueError("ASCD beta must be positive")
    cutoff = torch.log(
        torch.tensor(beta, dtype=logits.dtype, device=logits.device)
    )
    cutoff = cutoff + logits.max(dim=-1, keepdim=True).values
    combined = (1.0 + alpha) * logits - alpha * contrastive_logits
    return combined.masked_fill(logits < cutoff, -float("inf"))


try:
    from transformers.generation.utils import SampleEncoderDecoderOutput, SampleOutput
except ImportError:
    from transformers.generation.utils import (
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
    )

    SampleOutput = GenerateDecoderOnlyOutput
    SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput

def _isolated_branch_kwargs(model_kwargs):
    """Return a copy of ``model_kwargs`` with its own KV cache + cache_position.

    Transformers >=5 pre-instantiates a single ``DynamicCache`` and stores it in
    ``model_kwargs``. A plain ``dict.copy()`` is shallow, so the contrastive
    (neg/pos) branch would alias the SAME cache as the main branch: the two
    forwards then interleave their key/value writes into one cache (it grows by
    two positions per decode step) and the main branch ends up attending to the
    contrastive branch's polluted keys, producing degenerate, token-doubling
    output. Deep-copying the cache here keeps the branches fully independent.
    """
    branch = model_kwargs.copy()
    pkv = model_kwargs.get("past_key_values")
    if pkv is not None:
        branch["past_key_values"] = copy.deepcopy(pkv)
    cache_position = model_kwargs.get("cache_position")
    if cache_position is not None:
        branch["cache_position"] = cache_position.clone()
    return branch


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False

    model_kwargs_pos = _isolated_branch_kwargs(model_kwargs)
    model_kwargs_neg = _isolated_branch_kwargs(model_kwargs)

    t = 0
    total_overlapping_index_len = []
    while True:
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_only = model_kwargs.get("use_only")

        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if use_only:
            outputs, logits_cd = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            outputs, _ = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        if use_ritual or use_vcd or use_m3id or use_only:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs.get("images_pos") is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos, _ = self(
                    **model_inputs_pos,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :]

            elif model_kwargs.get("images_neg") is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg, _ = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg, _ = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]

            ritual_alpha_pos = model_kwargs.get("ritual_alpha_pos") if model_kwargs.get("ritual_alpha_pos") is not None else 3
            ritual_alpha_neg = model_kwargs.get("ritual_alpha_neg") if model_kwargs.get("ritual_alpha_neg") is not None else 1
            ritual_beta = model_kwargs.get("ritual_beta") if model_kwargs.get("ritual_beta") is not None else 0.1
            js_gamma = model_kwargs.get("js_gamma") if model_kwargs.get("js_gamma") is not None else 0.1

            cutoff = torch.log(torch.tensor(ritual_beta)) + next_token_logits.max(dim=-1, keepdim=True).values

            if use_ritual:
                diffs = next_token_logits + ritual_alpha_pos * next_token_logits_pos
            elif use_vcd:
                diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_neg
            elif use_m3id:
                gamma_t = torch.exp(torch.tensor(-0.02 * t))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg) * (1 - gamma_t) / gamma_t
                t += 1
            elif use_only:
                assert logits_cd is not None
                next_token_logits_cd = logits_cd[:, -1, :]
                tvd = torch.sum(torch.abs(
                    nn.functional.softmax(next_token_logits, dim=-1) -
                    nn.functional.softmax(next_token_logits_cd, dim=-1)
                ))
                total_overlapping_index_len.append(tvd.item())

                if tvd < js_gamma:
                    diffs = next_token_logits + ritual_alpha_pos * next_token_logits_cd
                else:
                    diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_cd

            logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            logits = logits_processor(input_ids, logits)
            logits = logits_warper(input_ids, logits)

            next_token_scores = logits
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_vcd or use_m3id:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids, decoder_attentions
    else:
        return input_ids, total_overlapping_index_len

def _llava_forward(self, model_inputs, use_only, output_attentions, output_hidden_states):
    """Run LLaVA forward; ONLY mode returns (output, logits_cd)."""
    if use_only:
        return self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
    out = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    if isinstance(out, tuple):
        return out[0], None
    return out, None


def _sample_llava(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    """Transformers >=5 sampling with local baselines and pinned ASCD."""
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    return_dict_in_generate = generation_config.return_dict_in_generate
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    this_peer_finished = False
    model_kwargs_pos = _isolated_branch_kwargs(model_kwargs)
    model_kwargs_neg = _isolated_branch_kwargs(model_kwargs)
    model_kwargs_ascd = (
        _isolated_branch_kwargs(model_kwargs)
        if model_kwargs.get("use_ascd")
        else None
    )
    t = 0
    total_overlapping_index_len = []

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_only = model_kwargs.get("use_only")
        use_ascd = model_kwargs.get("use_ascd")

        if use_ascd:
            set_ascd_branch(self.model, 0)

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs, logits_cd = _llava_forward(
            self, model_inputs, use_only, output_attentions, output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device,
        )

        if use_ritual or use_vcd or use_m3id or use_only or use_ascd:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs.get("images_pos") is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos, _ = _llava_forward(
                    self, model_inputs_pos, False, output_attentions, output_hidden_states,
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )
            elif model_kwargs.get("images_neg") is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg, _ = _llava_forward(
                    self, model_inputs_neg, False, output_attentions, output_hidden_states,
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg, _ = _llava_forward(
                    self, model_inputs_neg, False, output_attentions, output_hidden_states,
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )
            elif use_ascd:
                assert model_kwargs_ascd is not None
                model_inputs_ascd = self.prepare_inputs_for_generation(
                    input_ids, **model_kwargs_ascd
                )
                set_ascd_branch(self.model, 1)
                try:
                    outputs_ascd, _ = _llava_forward(
                        self,
                        model_inputs_ascd,
                        False,
                        output_attentions,
                        output_hidden_states,
                    )
                finally:
                    set_ascd_branch(self.model, 0)
                next_token_logits_ascd = outputs_ascd.logits[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )

            ritual_alpha_pos = model_kwargs.get("ritual_alpha_pos", 3)
            ritual_alpha_neg = model_kwargs.get("ritual_alpha_neg", 1)
            ritual_beta = model_kwargs.get("ritual_beta", 0.1)
            js_gamma = model_kwargs.get("js_gamma", 0.1)

            beta_safe = max(float(ritual_beta), 1e-8)
            cutoff = torch.log(torch.tensor(beta_safe, device=next_token_logits.device, dtype=next_token_logits.dtype))
            cutoff = cutoff + next_token_logits.max(dim=-1, keepdim=True).values

            if use_ritual:
                diffs = next_token_logits + ritual_alpha_pos * next_token_logits_pos
            elif use_vcd:
                diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_neg
            elif use_m3id:
                gamma_t = torch.exp(torch.tensor(-0.02 * t, device=next_token_logits.device))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg) * (1 - gamma_t) / gamma_t
                t += 1
            elif use_only:
                assert logits_cd is not None
                next_token_logits_cd = logits_cd[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )
                tvd = torch.sum(torch.abs(
                    nn.functional.softmax(next_token_logits, dim=-1) -
                    nn.functional.softmax(next_token_logits_cd, dim=-1)
                ))
                total_overlapping_index_len.append(tvd.item())
                if tvd < js_gamma:
                    diffs = next_token_logits + ritual_alpha_pos * next_token_logits_cd
                else:
                    diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_cd
            elif use_ascd:
                diffs = ascd_contrastive_logits(
                    next_token_logits,
                    next_token_logits_ascd,
                    alpha=float(model_kwargs.get("ascd_alpha", 1.0)),
                    beta=float(model_kwargs.get("ascd_beta", 0.1)),
                )

            if use_ascd:
                logits = diffs
            else:
                logits = diffs.masked_fill(
                    next_token_logits < cutoff, -float("inf")
                )
            next_token_scores = logits_processor(input_ids, logits)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate and output_scores:
            scores += (next_token_scores,)

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder,
            )
        if use_vcd or use_m3id:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder,
            )
        if use_ascd:
            assert model_kwargs_ascd is not None
            model_kwargs_ascd = self._update_model_kwargs_for_generation(
                outputs_ascd,
                model_kwargs_ascd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

        this_peer_finished = stopping_criteria(input_ids, scores)

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        return transformers.generation.utils.GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
        )
    return input_ids


_LAVA_EXTRA_KWARGS = {
    "images", "images_pos", "images_neg",
    "use_ritual", "use_vcd", "use_m3id", "use_only", "use_ascd",
    "enhance_layer_index",
    "ritual_alpha_pos", "ritual_alpha_neg", "ritual_beta", "js_gamma",
    "ascd_alpha", "ascd_beta",
}
_orig_llava_validate_model_kwargs = None


def _patched_llava_validate_model_kwargs(self, model_kwargs):
    filtered = {k: v for k, v in model_kwargs.items() if k not in _LAVA_EXTRA_KWARGS}
    return _orig_llava_validate_model_kwargs(self, filtered)


def evolve_only_sampling():
    gm = transformers.generation.utils.GenerationMixin
    gm._sample = _sample_llava
    if hasattr(gm, "sample"):
        gm.sample = sample

    global _orig_llava_validate_model_kwargs
    if _orig_llava_validate_model_kwargs is None:
        _orig_llava_validate_model_kwargs = gm._validate_model_kwargs
    gm._validate_model_kwargs = _patched_llava_validate_model_kwargs
