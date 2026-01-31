# Inference Server Enhancements for Latest Models

Summary of research (GitHub + web) on how to enhance this inference server for latest models and better performance. Current stack: **Starlette WebSocket + Hugging Face Transformers + bitsandbytes 4-bit + JSON-RPC streaming + tool calling**.

---

## Current project enhancements (no Docker)

The project is enhanced to run fully in-process with configurable model and attention backend:

- **Env-based config** – Copy `.env.example` to `.env` (or set env vars). `shared.py` loads `.env` via `python-dotenv`.
- **INFERENCE_MODEL_ID** – Hugging Face model id or local path (default: `./Qwen-14B`).
- **INFERENCE_ATTN_IMPLEMENTATION** – `sdpa` (default, built-in) or `flash_attention_2` (install `flash-attn` for GPU).
- **INFERENCE_BNB_COMPUTE_DTYPE** – `float16` or `bfloat16` for 4-bit quant compute dtype.
- **Validation** – `ATTN_IMPLEMENTATION` must be one of `sdpa`, `flash_attention_2`, `eager`; invalid value raises.

No Docker required; same single-process server with improved attention and config.

---

## 1. Stay on Transformers (In-Process) – Quick Wins

### 1.1 Flash Attention 2 / SDPA (implemented)
- **What**: `attn_implementation` is set from env in `shared.py` (`sdpa` or `flash_attention_2`).
- **Why**: Faster attention, less memory, better for long context.
- **Requires**: For `flash_attention_2`, `pip install flash-attn` (GPU + CUDA). SDPA is built into PyTorch.

### 1.2 Better Quantization Options
- **Current**: bitsandbytes 4-bit.
- **Alternatives**: Pre-quantized **AWQ** or **GPTQ** (often 1.2–1.7× faster, &lt;1% quality loss). Use `AutoModelForCausalLM.from_pretrained` with `quantization_config` for AWQ/GPTQ.
- **Latest models**: Prefer official pre-quantized variants (e.g. `Qwen/Qwen2.5-7B-Instruct-AWQ`) when available.

### 1.3 `transformers serve` (OpenAI-Compatible)
- **What**: Official CLI server with **continuous batching**, `/v1/chat/completions`, streaming, and experimental **tool calling / Responses API**.
- **Use case**: Evaluation, moderate load, no extra backend; same ecosystem as your current code.
- **Docs**: [Transformers serving](https://huggingface.co/docs/transformers/main/serving).

```bash
pip install "transformers[serving]"
transformers serve --quantization bnb-4bit --continuous-batching --attn-implementation sdpa
# Optional: --attn-implementation flash_attention_2
```

- You could **proxy** your existing WebSocket/JSON-RPC layer to `transformers serve`’s HTTP API instead of loading the model in-process.

### 1.4 Latest Model Families (Qwen, DeepSeek, etc.)
- **Qwen2.5 / Qwen3**: Use `Qwen2.5ForCausalLM` / official IDs; chat template and tool use are built-in.
- **DeepSeek-R1 / Distill**: Use `trust_remote_code=True` and official repo; check [DeepSeek](https://github.com/deepseek-ai) and [Hugging Face](https://huggingface.co/deepseek-ai) for recommended configs.
- **General**: Prefer `device_map="auto"`, `torch_dtype=torch.bfloat16` (or float16) when not quantizing, and the model’s native chat template.

---

## 2. Move to a Dedicated Inference Engine (High Throughput / Production)

If you need **higher throughput**, **continuous batching**, and **production-grade latency**, use a dedicated server and keep your app as a **client** (or thin proxy).

### 2.1 vLLM
- **Strengths**: PagedAttention, continuous batching, very high tokens/sec; OpenAI-compatible API; **tool calling** with `--enable-auto-tool-choice` and model-specific `--tool-call-parser`.
- **Docs**: [vLLM](https://docs.vllm.ai/), [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html), [tool calls](https://docs.vllm.ai/en/stable/getting_started/examples/openai_chat_completion_tool_calls_with_reasoning.html).

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes  # or qwen/mistral
```

- Your **Starlette WebSocket server** can call vLLM’s `/v1/chat/completions` (streaming) and map responses back to your JSON-RPC notifications.

### 2.2 Text Generation Inference (TGI)
- **Strengths**: Hugging Face ecosystem, **speculative decoding**, good single-request latency, Docker-first.
- **Docs**: [TGI](https://huggingface.co/docs/text-generation-inference), [streaming](https://huggingface.co/docs/text-generation-inference/conceptual/streaming), [Qwen](https://qwen.readthedocs.io/en/v2.0/deployment/tgi.html).

- Your app would send HTTP/SSE or use TGI’s OpenAI-compatible endpoint; WebSocket layer stays in your server.

### 2.3 SGLang
- **Strengths**: RadixAttention, prefix caching, strong for **complex prompts and agentic** workloads.
- **Project**: [SGLang](https://github.com/sgl-project/sglang).

### 2.4 Comparison (from research)
- **Throughput**: vLLM, SGLang, TensorRT-LLM &gt; TGI &gt; in-process Transformers.
- **Low latency / interactive**: TGI or vLLM.
- **Hugging Face integration**: TGI, then vLLM (OpenAI-compatible).
- **Complex prompts / agents**: SGLang or vLLM.

---

## 3. Architecture Pattern: OpenAI-Compatible Backend

A common approach is to keep your **WebSocket + JSON-RPC + tool-calling protocol** unchanged and **route generation to an OpenAI-compatible backend** (vLLM, TGI, or `transformers serve`):

1. **Backend** exposes `/v1/chat/completions` (and optionally `/v1/responses`) with streaming.
2. **Your server** (`server_starlette.py`) translates:
   - `GenerateRequestParams` + `HistoryMessage` → OpenAI-style `messages` (+ `tools` if supported).
   - Stream chunks → your `text_chunk` / `function_call_request` / `stream_end` notifications.
3. You can **switch backends** (e.g. from in-process Transformers to vLLM) without changing the client contract.

---

## 4. Concrete Checklist for *This* Repo

| Area | Action |
|------|--------|
| **Attention** | Add `attn_implementation="sdpa"` or `"flash_attention_2"` in `shared.py` if GPU allows. |
| **Quantization** | Consider AWQ/GPTQ or pre-quantized model IDs for speed; keep bitsandbytes as fallback. |
| **Model ID** | Prefer latest Qwen2.5 / Qwen3 or DeepSeek official IDs and `trust_remote_code` as required. |
| **Serving path** | Option A: Keep current in-process flow, add Flash/SDPA + better quantization. Option B: Introduce `transformers serve` or vLLM as backend and make Starlette a proxy. |
| **Tool calling** | If you move to vLLM: use `--enable-auto-tool-choice` and the right `--tool-call-parser`; map OpenAI tool_calls to your `function_call_request`. |
| **Observability** | Add metrics (latency, token throughput, errors) and structured logs for production. |

---

## 5. References

- [Transformers – Serving](https://huggingface.co/docs/transformers/main/serving) – `transformers serve`, continuous batching, Responses API.
- [vLLM – OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) – API surface.
- [vLLM – Tool calls (reasoning)](https://docs.vllm.ai/en/stable/getting_started/examples/openai_chat_completion_tool_calls_with_reasoning.html) – tool calling.
- [TGI – Streaming](https://huggingface.co/docs/text-generation-inference/conceptual/streaming) – streaming behavior.
- [Inference stacks comparison (vLLM, TGI, TensorRT, SGLang)](https://www.maniac.ai/blog/inference-stacks-vllm-tgi-tensorrt) – high-level tradeoffs.
- [Qwen2.5-Coder + TensorRT-LLM lookahead](https://developer.nvidia.com/blog/optimizing-qwen2-5-coder-throughput-with-nvidia-tensorrt-llm-lookahead-decoding/) – advanced decoding (NVIDIA stack).
