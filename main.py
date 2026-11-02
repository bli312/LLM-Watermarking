import argparse

from transformers import AutoConfig, AutoTokenizer

from watermark import (
    DEFAULT_MODEL_NAME,
    GreenListWatermark,
    WatermarkConfig,
    generate_completion,
    load_model_and_tokenizer,
)


MODEL = DEFAULT_MODEL_NAME
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_P = 0.95
WATERMARKED_ONLY = False

GAMMA = 0.25
DELTA = 3.0
SECRET_KEY = 15485863
Z_THRESHOLD = 4.0


def build_parser():
    parser = argparse.ArgumentParser(description="greenlist LLM watermark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--prompt", required=True)

    detect_parser = subparsers.add_parser("detect")
    input_group = detect_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text")
    input_group.add_argument("--file")

    return parser


def watermark_config():
    return WatermarkConfig(
        gamma=GAMMA,
        delta=DELTA,
        secret_key=SECRET_KEY,
        z_threshold=Z_THRESHOLD,
    )


def model_vocab_size(model_name):
    config = AutoConfig.from_pretrained(model_name)
    return int(config.vocab_size)


def print_detection(label, result):
    print(f"\n{label} detection")
    print(f"  scored tokens:   {result.total_tokens}")
    print(f"  green tokens:    {result.green_tokens}")
    print(f"  green fraction:  {result.green_fraction:.3f}")
    print(f"  expected:        {result.expected_fraction:.3f}")
    print(f"  z-score:         {result.z_score:.2f}")
    print(f"  one-sided p:     {result.p_value:.3g}")
    print(f"  watermarked:     {result.is_watermarked}")


def run_generate(prompt):
    model, tokenizer = load_model_and_tokenizer(MODEL)
    watermark = GreenListWatermark(
        vocab_size=int(model.config.vocab_size),
        config=watermark_config(),
    )

    if not WATERMARKED_ONLY:
        plain_text = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            watermark=None,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        print("\nUnwatermarked output ===")
        print(plain_text.strip())
        print_detection("Unwatermarked", watermark.score_text(plain_text, tokenizer))

    watermarked_text = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        watermark=watermark,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    print("\nWatermarked output ===")
    print(watermarked_text.strip())
    print_detection("Watermarked", watermark.score_text(watermarked_text, tokenizer))


def run_detect(text=None, file=None):
    if text is None:
        with open(file, encoding="utf-8") as f:
            text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    vocab_size = model_vocab_size(MODEL)
    watermark = GreenListWatermark(vocab_size=vocab_size, config=watermark_config())
    print_detection("Input", watermark.score_text(text, tokenizer))


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args.prompt)
    elif args.command == "detect":
        run_detect(text=args.text, file=args.file)


if __name__ == "__main__":
    main()
