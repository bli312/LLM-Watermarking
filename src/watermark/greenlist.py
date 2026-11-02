import hashlib
import math
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@dataclass(frozen=True)
class WatermarkConfig:
    gamma: float = 0.25
    delta: float = 3.0
    secret_key: int = 15485863
    z_threshold: float = 4.0
    cache_size: int = 4096


@dataclass(frozen=True)
class DetectionResult:
    total_tokens: int
    green_tokens: int
    green_fraction: float
    expected_fraction: float
    z_score: float
    p_value: float
    is_watermarked: bool


class GreenListWatermark:
    """
    chooses a psuedorandom greenlist of tokens, which get a logit boost in generation
    the same greenlist tokens are counted during detection
    see https://arxiv.org/pdf/2301.10226
    """

    def __init__(self, vocab_size, config=None):
        self.vocab_size = vocab_size
        self.config = config or WatermarkConfig()
        self.greenlist_size = max(1, int(round(self.vocab_size * self.config.gamma)))
        self._greenlist_cache = {}
        self._green_set_cache = {}

    def _get_token_seed(self, prev_token_id):
        data = f"{self.config.secret_key}:{prev_token_id}".encode("utf-8")
        digest = hashlib.sha256(data).digest()
        return int.from_bytes(digest[:8]) % (2**63 - 1)

    def _trim_cache(self):
        if len(self._greenlist_cache) < self.config.cache_size:
            return
        oldest_key = next(iter(self._greenlist_cache))
        self._greenlist_cache.pop(oldest_key, None)
        self._green_set_cache.pop(oldest_key, None)

    def greenlist_ids(self, prev_token_id):
        prev_token_id = int(prev_token_id)
        cached = self._greenlist_cache.get(prev_token_id)
        if cached is not None:
            return cached

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._get_token_seed(prev_token_id))
        permutation = torch.randperm(
            self.vocab_size, generator=generator, dtype=torch.long
        )
        green_ids = permutation[: self.greenlist_size]

        self._trim_cache()
        self._greenlist_cache[prev_token_id] = green_ids
        return green_ids

    def is_green_token(self, prev_token_id, token_id):
        prev_token_id = int(prev_token_id)
        green_set = self._green_set_cache.get(prev_token_id)
        if green_set is None:
            green_set = set(self.greenlist_ids(prev_token_id).tolist())
            self._green_set_cache[prev_token_id] = green_set
        return int(token_id) in green_set

    def score_token_ids(self, token_ids):
        token_ids = [int(token_id) for token_id in token_ids]
        if len(token_ids) < 2:
            return DetectionResult(
                total_tokens=0,
                green_tokens=0,
                green_fraction=0.0,
                expected_fraction=self.config.gamma,
                z_score=0.0,
                p_value=1.0,
                is_watermarked=False,
            )

        green_count = 0
        total = 0
        for prev_token_id, token_id in zip(token_ids[:-1], token_ids[1:]):
            if token_id < 0 or token_id >= self.vocab_size:
                continue
            total += 1
            if self.is_green_token(prev_token_id, token_id):
                green_count += 1

        if total == 0:
            z_score = 0.0
            p_value = 1.0
            green_fraction = 0.0
        else:
            # binomial distr: E[X] = np, V[X] = np(1-p)
            expected = total * self.config.gamma
            variance = total * self.config.gamma * (1.0 - self.config.gamma)
            z_score = (green_count - expected) / math.sqrt(variance)
            p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
            green_fraction = green_count / total

        return DetectionResult(
            total_tokens=total,
            green_tokens=green_count,
            green_fraction=green_fraction,
            expected_fraction=self.config.gamma,
            z_score=z_score,
            p_value=p_value,
            is_watermarked=z_score >= self.config.z_threshold,
        )

    def score_text(self, text, tokenizer):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return self.score_token_ids(token_ids)


class GreenListLogitsProcessor(LogitsProcessor):
    def __init__(self, watermark):
        self.watermark = watermark

    def __call__(self, input_ids, scores):
        # input_ids (torch.LongTensor of shape (batch_size, sequence_length))
        # scores (torch.FloatTensor of shape (batch_size, config.vocab_size))
        for batch_index in range(scores.shape[0]):
            prev_token_id = int(input_ids[batch_index, -1].item())
            green_ids = self.watermark.greenlist_ids(prev_token_id).to(scores.device)
            scores[batch_index, green_ids] += self.watermark.config.delta
        return scores


def load_model_and_tokenizer(model_name=DEFAULT_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _format_prompt(tokenizer, prompt):
    if tokenizer.chat_template is None:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_completion(
    model,
    tokenizer,
    prompt,
    watermark,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
):

    formatted_prompt = _format_prompt(tokenizer, prompt)
    # tokenizer gives "input_ids" and "attention_mask"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}

    logits_processor = LogitsProcessorList()
    if watermark is not None:
        logits_processor.append(GreenListLogitsProcessor(watermark))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0, prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
