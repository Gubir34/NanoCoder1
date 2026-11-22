# agentcoder-worker-1.5b

LoRA adapter (r=16, all attention + MLP projections) for `Qwen/Qwen2.5-Coder-1.5B-Instruct`,
plus retrained embedding rows for `<tool_call>` / `</tool_call>`. Same 1364-sample mix as
`agentcoder-worker-0.5b` (agent trajectories, MBPP repo tasks, distilled MBPP answers), 1 epoch with
`finetune/train.py --grad-ckpt` on a 4-core CPU (~4.5 h, run in 110-minute chunks with `--resume`).

Stored in fp16 (trained in fp32).

| | agentic (90 tasks) | MBPP test (495) |
|---|---|---|
| base 1.5B | 2/90 | 266 (53.7%) |
| this adapter | 85/90 | 273 (55.2%) |

    agentcoder-server --model agentcoder-worker:1.5b=Qwen/Qwen2.5-Coder-1.5B-Instruct+finetune/adapters/agentcoder-worker-1.5b
    agentcoder -w medium=agentcoder-worker:1.5b
