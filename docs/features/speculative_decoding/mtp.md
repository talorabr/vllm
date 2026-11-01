# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## DeepSeek V4 Checkpoint MTP Configuration

An optional `mtp_config` object in the checkpoint's Hugging Face `config.json`
provides overrides for the draft HF configuration in the `deepseek_v4` /
`deepseek_v41` conversion branch. These fields are copied before deriving the
draft's layer count and architecture. The target configuration is not modified,
and other model families retain their existing configuration handling.

Fields replace the corresponding draft fields; nested values are copied, not
recursively merged. Model-specific architecture conversion and explicit runtime
overrides still take precedence over these checkpoint defaults. The separate
Transformers modeling-backend conversion path is unchanged.

`mtp_config` must be a JSON object. There is no field allowlist in this merge:
the model remains responsible for interpreting and
validating its settings. An arbitrary field is not necessarily supported just
because it can be loaded. Overrides must also be compatible with the checkpoint
weights and the model implementation.

This does not enable classic MTP on DeepSeek V4.1, which uses DSpark.

### DeepSeek V4 MTP-only Sliding Window

On the NVIDIA DeepSeek V4 MTP path, set `mtp_config.sliding_window` in the
checkpoint's Hugging Face `config.json` to override the MTP window independently
of the target:

```json
{
  "sliding_window": 128,
  "mtp_config": {
    "sliding_window": 8192
  }
}
```

This example keeps the target window at 128 and uses 8192 for MTP attention,
KV-cache retention, and attention metadata. When using a separate MTP checkpoint,
put `mtp_config` in that checkpoint's config. Without an override, the existing
window behavior is unchanged. Use this setting with `"method": "mtp"`; it is not
a DSpark configuration option.

DeepSeek V4's MTP sliding window must be a positive integer. Changing the window
at inference does not retrain the MTP weights and may change acceptance rates.

## Offline Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
- When the MTP head shares the target's unquantized lm_head (most MTP models),
  `"draft_lm_head_quantization": "fp8"` or `"nvfp4"` in `--speculative-config`
  gives the drafter a quantized copy of that head (Model Runner V2). The target
  still verifies with its own head, so outputs are unchanged; only the
  acceptance length can move. It helps most on bandwidth-bound GPUs with a
  large vocabulary. The copy costs about one byte (FP8) or 0.56 bytes (NVFP4)
  per lm_head weight, e.g. 0.6 or 0.35 GiB for a 248k x 2560 head; it is
  counted in the model weights, so with `--kv-cache-memory-bytes` leave room
  for it.
