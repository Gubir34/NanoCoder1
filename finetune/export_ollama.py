"""Merge an AgentCoder LoRA adapter into its base model, convert it to GGUF and import it into Ollama.

    python finetune/export_ollama.py --adapter finetune/adapters/agentcoder-worker-0.5b \
        --base Qwen/Qwen2.5-Coder-0.5B-Instruct --name agentcoder-worker:0.5b --llama-cpp ~/llama.cpp

Then: agentcoder -w small=agentcoder-worker:0.5b

Needs a llama.cpp checkout for convert_hf_to_gguf.py (pip install gguf sentencepiece). The GGUF carries the
HF chat template, and recent Ollama renders it as is. That matters: Ollama's own Go templates print tool
schemas and past arguments with keys sorted alphabetically (new_string, old_string, path), unlike training,
and the worker then forgot `path` in edit_file calls (7/10 with 4-5 tool errors vs 9/10 and none).
The GGUF also works directly with llama-server --jinja.
"""
import argparse
import os
import shutil
import subprocess
import sys

# Only for old Ollama versions that cannot render the GGUF's jinja template (--legacy-template).
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
    ap.add_argument("--name", default="agentcoder-worker:0.5b")
    ap.add_argument("--out", help="where to write the merged model (default: build/<name>)")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp")),
                    help="llama.cpp checkout with convert_hf_to_gguf.py")
    ap.add_argument("--outtype", default="q8_0", help="GGUF type: q8_0, f16, bf16, ...")
    ap.add_argument("--legacy-template", action="store_true",
                    help="write a Go TEMPLATE for old Ollama versions (tool calls work worse)")
    args = ap.parse_args()
    out = args.out or os.path.join("build", args.name.replace(":", "-"))
    convert = os.path.join(args.llama_cpp, "convert_hf_to_gguf.py")
    if not os.path.exists(convert):
        sys.exit(f"{convert} not found; clone llama.cpp and pass --llama-cpp")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"merging {args.adapter} into {args.base} ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)

    gguf = f"model-{args.outtype}.gguf"
    subprocess.run([sys.executable, convert, out, "--outtype", args.outtype, "--outfile", os.path.join(out, gguf)],
                   check=True)
    with open(os.path.join(out, "Modelfile"), "w") as fh:
        fh.write(f"FROM ./{gguf}\n")
        if args.legacy_template:
            fh.write(f'TEMPLATE """{TEMPLATE}"""\n')
        # num_predict: a guard against a runaway reply eating the whole context
        fh.write('PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|endoftext|>"\n'
                 "PARAMETER temperature 0.2\nPARAMETER num_ctx 16384\nPARAMETER num_predict 4096\n")
    print(f"wrote {out}/{gguf} and {out}/Modelfile")

    if not shutil.which("ollama"):
        print(f"ollama not found. Later run:  cd {out} && ollama create {args.name} -f Modelfile")
        return
    subprocess.run(["ollama", "create", args.name, "-f", "Modelfile"], cwd=out, check=True)
    print(f"done. Use it with:  agentcoder -w small={args.name}")


if __name__ == "__main__":
    main()
