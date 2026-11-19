# agentcoder-worker-0.5b

LoRA adapter (r=16, all attention + MLP projections) for `Qwen/Qwen2.5-Coder-0.5B-Instruct`,
plus retrained embedding rows for `<tool_call>` / `</tool_call>`. Trained by `finetune/train.py`
on 800 synthetic agent trajectories from `finetune/gen_data.py` (1 epoch, CPU, ~64 min).

    agentcoder-server --model agentcoder-worker:0.5b=Qwen/Qwen2.5-Coder-0.5B-Instruct+finetune/adapters/agentcoder-worker-0.5b
    agentcoder -w small=agentcoder-worker:0.5b

Or merge it into Ollama with `finetune/export_ollama.py`.
