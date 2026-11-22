# agentcoder-worker-0.5b

LoRA adapter (r=16, all attention + MLP projections) for `Qwen/Qwen2.5-Coder-0.5B-Instruct`,
plus retrained embedding rows for `<tool_call>` / `</tool_call>`. Trained by `finetune/train.py`
(1 epoch, CPU, ~113 min) on 1364 samples:

- 800 synthetic agent trajectories (`finetune/gen_data.py`)
- 280 "add a function to a repo and run its tests" trajectories built from MBPP train problems (`finetune/code_data.py`)
- 284 MBPP train answers with reasoning from a 1.5B teacher, kept only if their code passes the tests (`finetune/distill_code.py`)

Stored in fp16 (trained in fp32; the rounding error is ~0.05%, well below the bf16 the server runs in).

    agentcoder-server --model agentcoder-worker:0.5b=Qwen/Qwen2.5-Coder-0.5B-Instruct+finetune/adapters/agentcoder-worker-0.5b
    agentcoder -w small=agentcoder-worker:0.5b

Or merge it into Ollama with `finetune/export_ollama.py`.
