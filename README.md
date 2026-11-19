# ✻ AgentCoder

Terminalde çalışan, **tamamen yerel**, minicik bir kodlama ajanı (eski adıyla nanocoder). Tek Python dosyası, **sıfır bağımlılık**.
Dosya okur, arar, düzenler, komut çalıştırır. Bunları yapmadan önce sana sorar.

```
╭──────────────────────────────────────────────────────────╮
│ ✻ AgentCoder v0.1.0                                       │
│   model  qwen2.5-coder:7b                                │
╰──────────────────────────────────────────────────────────╯
❯ app.py'deki bug'ı düzelt
● Önce bir bakayım.
● Read(app.py)
  ⎿  Read 4 lines
● Edit(app.py)
  ╭─ Edit app.py
  │ -    return a - b
  │ +    return a + b
  ╰─ Allow? [y]es · [a]lways for Edit · [n]o · or type what to do instead
```

## Özellikler

- **Yerel model, en hızlı altyapıda**: makinedeki Ollama / llama.cpp / vLLM / SGLang / LM Studio sunucularını bulur, her model için en hızlısını ölçüp seçer
- **Tool'lar**: `read_file`, `write_file`, `edit_file`, `bash`, `glob`, `grep`
- **Onay sistemi**: düzenleme ve komutlardan önce renkli diff önizlemesi gösterir. Cevaplar: `y` / `a` (hep izin ver) / `n`. İstersen düz yazıyla "şunu yap" diyerek yönlendirebilirsin
- **Streaming markdown**: başlıklar, listeler, **bold**, `kod`, kod blokları
- **Düşünen modeller**: `<think>` blokları gizlenir ("✻ Thought for 12s"), `/think` ile açılır
- **Native tool desteği olmayan modeller**: otomatik olarak metin tabanlı `<tool_call>` moduna geçer
- **Akıllı edit**: boşluk farklarını tolere eder. Eşleşme bulamazsa modele en yakın eşleşmeyi gösterir, model de kendini düzeltir
- **Otomatik compaction**: context dolunca konuşmayı özetler
- `@dosya` ile dosya ekleme, `!komut` ile shell, proje notları için `AGENTCODER.md` (`AGENTS.md`/`CLAUDE.md` da okunur)
- Ctrl+C her zaman güvenli, konuşma geçmişi bozulmaz
- **Prompt düzenleyici**: ayrı bir model mesajını netleştirip "bunları yapma" listesiyle şefe iletir
- **Ajan ekibi**: büyük bir "şef" model planlar, küçük modellere iş dağıtır (aşağıda)
- **Fine-tune edilmiş işçi modeller**: agentcoder'ın tool akışına özel eğitilmiş LoRA adapter'ları (aşağıda)

## Kurulum

```bash
# 1) Ollama kur: https://ollama.com, sonra bir model çek
ollama pull qwen2.5-coder:7b

# 2) agentcoder'ı kur
pipx install git+https://github.com/gubir34/NanoCoder1
# ya da kurmadan çalıştır:
python3 agentcoder.py
```

> ⚠️ **Context uzunluğu**: Ollama varsayılan olarak küçük bir context kullanıyor ve ajanlar bunu çabuk doldurur.
> Ollama'yı şöyle başlat: `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`

## Kullanım

```bash
agentcoder                          # interaktif mod
agentcoder "testleri çalıştır ve kırık olanı düzelt"   # tek seferlik
agentcoder -m qwen3:8b              # başka model
agentcoder --yolo                   # hiçbir şey sormaz, cesursan
agentcoder --base-url http://localhost:1234/v1   # LM Studio
```

| Ortam değişkeni       | Varsayılan                   |
|-----------------------|------------------------------|
| `AGENTCODER_MODEL`     | `qwen2.5-coder:7b`           |
| `AGENTCODER_BASE_URL`  | `auto` (en hızlı yerel sunucu) |
| `AGENTCODER_API_KEY`   | boş                          |
| `AGENTCODER_CTX`       | `32768`                      |

### Altyapı seçimi (otomatik)

`--base-url` verilmezse AgentCoder açılışta yerel sunucuları tarar:

| Altyapı | Port |
|---|---|
| Ollama | 11434 |
| llama.cpp (`llama-server`) | 8080–8083 |
| vLLM | 8000 |
| SGLang | 30000 |
| LM Studio | 1234 |

İsimleri farklı da olsa (`qwen2.5-coder:7b`, `Qwen/Qwen2.5-Coder-7B-Instruct`, `qwen2.5-coder-7b-instruct-q4_k_m.gguf`)
aynı modeli tanır. Bir model birden fazla sunucudaysa kısa bir ölçüm yapıp **en hızlısını** kullanır.
Düzenleyici, şef ve her işçi için ayrı ayrı karar verir. Ölçümler `~/.agentcoder/bench.json` dosyasında bir hafta saklanır.

```bash
agentcoder --rebench                                # yeniden ölç
agentcoder --base-url http://localhost:8080/v1      # otomatiği kapat, tek sunucu kullan
AGENTCODER_BACKENDS="gpu=http://10.0.0.5:8000/v1,ollama=http://127.0.0.1:11434/v1" agentcoder   # kendi listen
```

Oturum içinde `/backends` sunucuları ve ölçüm sonuçlarını gösterir.

### Komutlar

| Komut | Ne yapar |
|---|---|
| `/help` | yardım |
| `/clear` | konuşmayı sıfırlar |
| `/compact` | konuşmayı özetleyip context'i boşaltır |
| `/init` | projeyi inceleyip `AGENTCODER.md` yazar |
| `/model [isim]`, `/models` | model gösterir, değiştirir, listeler |
| `/yolo` | otomatik onayı açar/kapatır |
| `/think` | modelin düşüncelerini gösterir/gizler |
| `/tokens` | context kullanımını gösterir |
| `/team` | düzenleyiciyi, şefi ve işçileri gösterir |
| `/refine` | prompt düzenleyiciyi açar/kapatır |
| `/backends` | bulunan sunucuları, modelleri ve hızlarını gösterir |
| `!ls -la` | shell komutu çalıştırır, çıktısı bir sonraki mesajla modele gider |
| `@src/main.py` | dosyayı mesaja ekler |

## Akış

```
sen ──▶ düzenleyici (qwen3.5:9b) ──brief──▶ şef (qwen2.5-coder:7b) ──task──▶ işçiler (0.5b / 1.5b / 3b)
```

## Prompt düzenleyici

Her mesajın önce düzenleyici modele gider. Düzenleyici mesajı şu formatta bir brief'e çevirir:

```
GOAL: ne isteniyor
DETAILS: somut gereksinimler, tahminler "(assumed)" diye işaretli
DO NOT: bu iş için yapılmaması gerekenler (alakasız dosyalara dokunma, bağımlılık ekleme...)
DONE WHEN: işin bittiği nasıl anlaşılır
REPLY LANGUAGE: hangi dilde cevap verilecek
```

Brief, orijinal mesajının altına eklenip şefe gider. Çelişki olursa senin yazdığın kazanır.
"evet", "devam" gibi kısa mesajlar düzenleyiciye uğramadan geçer.

```bash
ollama pull qwen3.5:9b
agentcoder -r gemma2:9b       # başka bir düzenleyici
agentcoder --no-refiner       # düzenleyici kapalı
```

Oturum içinde `/refine` ile açıp kapatabilirsin.

## Ajan ekibi (orkestratör + sub-agent'lar)

Ana model **şef** olarak çalışır ve `task` tool'u ile küçük modellere iş paslar. Küçük modeller kendi
tool döngülerini çalıştırıp şefe kısa bir rapor döner. Şef onların tool çıktılarını görmez, sadece raporu görür.
Böylece şefin context'i şişmez.

| Rol | Varsayılan model | Yetki |
|---|---|---|
| şef (`-m`) | `qwen2.5-coder:7b` | her şey + iş dağıtma |
| `small` | `qwen2.5-coder:0.5b` | **sadece okuma** (arama, okuma, özetleme) |
| `medium` | `qwen2.5-coder:1.5b` | okuma + düzenleme + komut |
| `large` | `qwen2.5-coder:3b` | okuma + düzenleme + komut |

```bash
for m in 7b 0.5b 1.5b 3b; do ollama pull qwen2.5-coder:$m; done
OLLAMA_MAX_LOADED_MODELS=4 OLLAMA_CONTEXT_LENGTH=32768 ollama serve

agentcoder -w small=qwen3:0.6b -w medium=qwen3:1.7b          # kendi ekibini kur
agentcoder -w big=llama3:8b@http://localhost:1234/v1          # başka sunucudaki model
agentcoder --no-workers                                       # tek ajan
```

- Sunucuda olmayan işçi modelleri başlangıçta otomatik devre dışı kalır
- Alt ajanların düzenleme ve komutları da aynı onay ekranından geçer. `n` dersen hem işçi hem şef durur
- Alt ajanlar başka ajan çağıramaz, yani sonsuz delegasyon zinciri yok
- `/team` ekibi gösterir. Ortam değişkeni: `AGENTCODER_WORKERS="small=...,medium=..."`

## Model önerileri

Tool calling'i iyi olan modeller daha iyi sonuç verir:

- `qwen2.5-coder:7b` / `14b`: hızlı, düzgün
- `qwen3:8b`, `qwen3-coder:30b`: daha akıllı, daha aç
- 3B altı modeller de çalışır ama sonuç tam bir piyango

## Lisans

MIT
