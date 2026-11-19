"""Merge a nanocoder LoRA adapter into its base model and import it into Ollama.

    python finetune/export_ollama.py --adapter finetune/adapters/nanocoder-worker-0.5b \
        --base Qwen/Qwen2.5-Coder-0.5B-Instruct --name nanocoder-worker:0.5b

Then: nanocoder -w small=nanocoder-worker:0.5b
"""
import argparse
import os
import shutil
import subprocess

# Mirrors the HF Qwen2.5 chat template the adapter was trained with (assistant text + tool calls together).
TEMPLATE = r'''{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}{{ .System }}{{ else }}You are a helpful assistant.{{ end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{- if .Content }}
{{ .Content }}
{{- end }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--name", default="nanocoder-worker:0.5b")
    ap.add_argument("--out", help="where to write the merged model (default: build/<name>)")
    ap.add_argument("--quantize", default="q8_0", help="ollama --quantize level, or 'none'")
    args = ap.parse_args()
    out = args.out or os.path.join("build", args.name.replace(":", "-"))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"merging {args.adapter} into {args.base} ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)

    with open(os.path.join(out, "Modelfile"), "w") as fh:
        fh.write(f'FROM .\nTEMPLATE """{TEMPLATE}"""\n'
                 'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|endoftext|>"\n'
                 "PARAMETER temperature 0.2\nPARAMETER num_ctx 16384\n")
    print(f"wrote {out}/Modelfile")

    if not shutil.which("ollama"):
        print(f"ollama not found. Later run:  cd {out} && ollama create {args.name} -f Modelfile")
        return
    cmd = ["ollama", "create", args.name, "-f", "Modelfile"]
    if args.quantize != "none":
        cmd += ["--quantize", args.quantize]
    subprocess.run(cmd, cwd=out, check=True)
    print(f"done. Use it with:  nanocoder -w small={args.name}")


if __name__ == "__main__":
    main()
