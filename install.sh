#!/usr/bin/env bash
# AgentCoder installer.
#
#   curl -fsSL https://raw.githubusercontent.com/gubir34/NanoCoder1/main/install.sh | bash
#
# What it does (each step asks first when a terminal is attached):
#   1. installs the `agentcoder` command from GitHub (pipx, uv or a private venv; no root needed)
#   2. installs Ollama if it is missing
#   3. pulls the orchestrator / refiner model, picked by how much RAM the machine has
#   4. optionally builds the fine-tuned workers (agentcoder-worker:0.5b / :1.5b) and imports them into Ollama
#   5. writes the choices to ~/.agentcoder/config
#
# Non-interactive knobs (env vars): AGENTCODER_REF (branch/tag, default main),
#   AGENTCODER_INSTALL_OLLAMA, AGENTCODER_PULL, AGENTCODER_BUILD_WORKERS = y|n,
#   AGENTCODER_CHEF (model name, overrides the RAM-based pick).
#
# Everything runs inside main(), so bash has read the whole file before anything executes:
# with `curl | bash` a child command reading stdin would otherwise eat the rest of the script.

main() {
    set -euo pipefail
    REPO="gubir34/NanoCoder1"
    REF="${AGENTCODER_REF:-main}"
    HOME_DIR="$HOME/.agentcoder"
    BIN_DIR="$HOME/.local/bin"
    TARBALL="https://github.com/$REPO/archive/$REF.tar.gz"
    mkdir -p "$HOME_DIR"

    say "AgentCoder kurulumu ($REPO@$REF)"
    find_python
    install_cli
    setup_ollama
    pick_models
    build_workers
    write_config
    finish
}

# ─────────────────────────────── helpers ───────────────────────────────

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'
else B=; G=; Y=; R=; D=; N=; fi
say()  { printf '%s●%s %s\n' "$G" "$N" "$*"; }
info() { printf '  %s%s%s\n' "$D" "$*" "$N"; }
warn() { printf '%s!%s %s\n' "$Y" "$N" "$*" >&2; }
die()  { printf '%s✗%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ask VAR "question" default(y|n): the env var wins, then the terminal, then the default
ask() {
    local var=$1 q=$2 def=$3 ans=""
    ans="${!var:-}"
    if [ -z "$ans" ] && (exec </dev/tty) 2>/dev/null; then
        local hint="y/N"; [ "$def" = y ] && hint="Y/n"
        printf '%s?%s %s [%s] ' "$B" "$N" "$q" "$hint" >/dev/tty
        read -r ans </dev/tty || ans=""
    fi
    case "${ans:-$def}" in [yYeE]*) return 0 ;; *) return 1 ;; esac
}

ram_gb() {
    if [ -r /proc/meminfo ]; then awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo
    elif have sysctl; then echo $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
    else echo 0; fi
}

# ─────────────────────────────── steps ─────────────────────────────────

find_python() {
    PY=""
    for p in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
        if have "$p" && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
            PY=$(command -v "$p"); break
        fi
    done
    [ -n "$PY" ] || die "Python 3.9+ bulunamadı. Önce Python kur (ör. sudo apt install python3 python3-venv)."
    info "python: $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"
}

install_cli() {
    say "agentcoder komutu kuruluyor"
    if have pipx; then
        pipx install --force "$TARBALL" >/dev/null </dev/null || die "pipx install başarısız"
        info "pipx ile kuruldu"
    elif have uv; then
        uv tool install -q --force --python "$PY" "$TARBALL" </dev/null || die "uv tool install başarısız"
        info "uv tool ile kuruldu"
    else
        # a private venv: works on PEP 668 systems (Debian/Ubuntu) without --break-system-packages
        if ! "$PY" -m venv --clear "$HOME_DIR/venv" >/dev/null 2>&1; then
            die "venv oluşturulamadı. Debian/Ubuntu'da: sudo apt install python3-venv (ya da pipx kur)"
        fi
        "$HOME_DIR/venv/bin/python" -m pip install -q --upgrade "$TARBALL" </dev/null \
            || die "pip install başarısız"
        mkdir -p "$BIN_DIR"
        ln -sf "$HOME_DIR/venv/bin/agentcoder" "$BIN_DIR/agentcoder"
        ln -sf "$HOME_DIR/venv/bin/agentcoder-server" "$BIN_DIR/agentcoder-server"
        info "$HOME_DIR/venv içine kuruldu, $BIN_DIR/agentcoder"
    fi
}

ollama_up() { curl -fsS -m 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; }

setup_ollama() {
    OLLAMA_OK=0
    if ! have ollama; then
        if ask AGENTCODER_INSTALL_OLLAMA "Ollama yok. Kurayım mı? (Linux'ta sudo ister)" y; then
            case "$(uname -s)" in
                Linux)  curl -fsSL https://ollama.com/install.sh </dev/null | sh || die "Ollama kurulamadı" ;;
                Darwin) if have brew; then brew install ollama </dev/null; brew services start ollama </dev/null || true
                        else die "macOS'ta Ollama'yı https://ollama.com/download adresinden kur, sonra bu scripti tekrar çalıştır"; fi ;;
                *)      die "Bu sistemde Ollama'yı elle kur: https://ollama.com/download" ;;
            esac
            context_override
        else
            warn "Ollama kurulmadı. AgentCoder llama.cpp / vLLM / LM Studio sunucularını da otomatik bulur."
            return
        fi
    fi
    if ! ollama_up; then
        info "Ollama sunucusu kapalı, başlatıyorum"
        OLLAMA_CONTEXT_LENGTH=32768 nohup ollama serve >"$HOME_DIR/ollama.log" 2>&1 </dev/null &
        for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
    fi
    if ollama_up; then OLLAMA_OK=1; info "Ollama çalışıyor"
    else warn "Ollama sunucusuna ulaşılamadı ($HOME_DIR/ollama.log). Model indirme adımı atlanıyor."; fi
}

# Ollama's default context (a few thousand tokens) is far too small for an agent: raise it for the service
context_override() {
    if [ "$(uname -s)" = Linux ] && have systemctl && systemctl cat ollama >/dev/null 2>&1; then
        local sudo=""; [ "$(id -u)" = 0 ] || sudo="sudo"
        $sudo mkdir -p /etc/systemd/system/ollama.service.d
        printf '[Service]\nEnvironment="OLLAMA_CONTEXT_LENGTH=32768"\n' \
            | $sudo tee /etc/systemd/system/ollama.service.d/agentcoder.conf >/dev/null
        $sudo systemctl daemon-reload && $sudo systemctl restart ollama
        info "Ollama servisinin context'i 32768'e ayarlandı"
    fi
}

pick_models() {
    local ram; ram=$(ram_gb)
    if [ -n "${AGENTCODER_CHEF:-}" ]; then CHEF=$AGENTCODER_CHEF
    elif [ "$ram" -ge 12 ]; then CHEF="qwen3.5:9b"      # most reliable tool-calling chef we tested on CPU
    else CHEF="qwen2.5-coder:7b"; fi
    REFINER=$CHEF   # one model does both jobs: one download, nothing extra in RAM
    info "RAM: ${ram}GB → şef ve düzenleyici: $CHEF"
    [ "$ram" -ge 12 ] || [ -n "${AGENTCODER_CHEF:-}" ] || warn "12GB'tan az RAM: 7B şef yavaş ve daha az güvenilir olur"
    [ "$OLLAMA_OK" = 1 ] || return 0
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$CHEF"; then info "$CHEF zaten var"
    elif ask AGENTCODER_PULL "$CHEF indirilsin mi? (~5-7GB)" y; then ollama pull "$CHEF" </dev/null
    else warn "Model indirilmedi: sonra 'ollama pull $CHEF'"; fi
}

build_workers() {
    WORKERS=""
    [ "$OLLAMA_OK" = 1 ] || return 0
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "agentcoder-worker:1.5b"; then
        info "fine-tune'lu işçiler zaten kurulu"
        WORKERS="small=agentcoder-worker:0.5b,medium=agentcoder-worker:1.5b"; return
    fi
    ask AGENTCODER_BUILD_WORKERS "Fine-tune'lu işçiler (0.5B + 1.5B) derlensin mi? (~4GB indirme, 10-20 dk; atlarsan şef tek başına çalışır)" n \
        || return 0
    have git || die "işçileri derlemek için git lazım"
    local src="$HOME_DIR/src" llama="$HOME_DIR/llama.cpp" venv="$HOME_DIR/build-venv"
    say "işçiler derleniyor (merge → GGUF → ollama create)"
    rm -rf "$src"; git clone -q --depth 1 --branch "$REF" "https://github.com/$REPO" "$src" </dev/null
    [ -d "$llama" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$llama" </dev/null
    "$PY" -m venv --clear "$venv" >/dev/null || die "venv oluşturulamadı"
    local torch_idx=()
    [ "$(uname -s)" = Linux ] && torch_idx=(--index-url https://download.pytorch.org/whl/cpu)
    "$venv/bin/pip" install -q torch "${torch_idx[@]}" </dev/null
    "$venv/bin/pip" install -q transformers peft safetensors sentencepiece protobuf "$llama/gguf-py" </dev/null
    (
        cd "$HOME_DIR"
        "$venv/bin/python" "$src/finetune/export_ollama.py" --llama-cpp "$llama" --out "$HOME_DIR/build/w05" \
            --adapter "$src/finetune/adapters/agentcoder-worker-0.5b" \
            --base Qwen/Qwen2.5-Coder-0.5B-Instruct --name agentcoder-worker:0.5b </dev/null
        "$venv/bin/python" "$src/finetune/export_ollama.py" --llama-cpp "$llama" --out "$HOME_DIR/build/w15" \
            --adapter "$src/finetune/adapters/agentcoder-worker-1.5b" \
            --base Qwen/Qwen2.5-Coder-1.5B-Instruct --name agentcoder-worker:1.5b </dev/null
    ) || die "işçi derlemesi başarısız"
    rm -rf "$HOME_DIR/build" "$venv"   # Ollama keeps its own copy; torch & merged weights are ~5GB
    WORKERS="small=agentcoder-worker:0.5b,medium=agentcoder-worker:1.5b"
    info "işçiler Ollama'ya eklendi"
}

write_config() {
    local cfg="$HOME_DIR/config"
    {
        echo "# written by install.sh; environment variables override these"
        echo "AGENTCODER_MODEL=$CHEF"
        echo "AGENTCODER_REFINER=$REFINER"
        case "$CHEF" in qwen3*) echo "AGENTCODER_NO_THINK=1" ;; esac   # thinking costs minutes per step on CPU
        [ -n "$WORKERS" ] && echo "AGENTCODER_WORKERS=$WORKERS"
    } >"$cfg"
    info "ayarlar: $cfg"
}

finish() {
    echo
    say "${B}Kurulum bitti.${N} Bir proje klasöründe çalıştır:  ${B}agentcoder${N}"
    if ! have agentcoder; then
        case ":$PATH:" in *":$BIN_DIR:"*) ;; *)
            warn "$BIN_DIR PATH'te değil. Şunu ~/.bashrc ya da ~/.zshrc'ye ekle:"
            printf '    export PATH="%s:$PATH"\n' "$BIN_DIR" ;;
        esac
    fi
    [ -n "$WORKERS" ] || info "işçisiz kuruldu; sonra eklemek için: AGENTCODER_BUILD_WORKERS=y ile tekrar çalıştır"
}

main "$@"
