# ✻ nanocoder

Terminalde çalışan, **tamamen yerel**, minicik bir kodlama ajanı. Tek Python dosyası, **sıfır bağımlılık**.
Dosya okur, arar, düzenler, komut çalıştırır. Bunları yapmadan önce sana sorar.

```
╭──────────────────────────────────────────────────────────╮
│ ✻ nanocoder v0.1.0                                       │
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

- **Yerel model**: Ollama, LM Studio, llama.cpp, vLLM, yani OpenAI uyumlu her sunucuyla çalışır
- **Tool'lar**: `read_file`, `write_file`, `edit_file`, `bash`, `glob`, `grep`
- **Onay sistemi**: düzenleme ve komutlardan önce renkli diff önizlemesi gösterir. Cevaplar: `y` / `a` (hep izin ver) / `n`. İstersen düz yazıyla "şunu yap" diyerek yönlendirebilirsin
- **Streaming markdown**: başlıklar, listeler, **bold**, `kod`, kod blokları
- **Düşünen modeller**: `<think>` blokları gizlenir ("✻ Thought for 12s"), `/think` ile açılır
- **Native tool desteği olmayan modeller**: otomatik olarak metin tabanlı `<tool_call>` moduna geçer
- **Akıllı edit**: boşluk farklarını tolere eder. Eşleşme bulamazsa modele en yakın eşleşmeyi gösterir, model de kendini düzeltir
- **Otomatik compaction**: context dolunca konuşmayı özetler
- `@dosya` ile dosya ekleme, `!komut` ile shell, proje notları için `NANOCODER.md` (`AGENTS.md`/`CLAUDE.md` da okunur)
- Ctrl+C her zaman güvenli, konuşma geçmişi bozulmaz
- **Ajan ekibi**: büyük bir "şef" model planlar, küçük modellere iş dağıtır (aşağıda)

## Kurulum

```bash
# 1) Ollama kur: https://ollama.com, sonra bir model çek
ollama pull qwen2.5-coder:7b

# 2) nanocoder'ı kur
pipx install git+https://github.com/gubir34/nanocoder1
# ya da kurmadan çalıştır:
python3 nanocoder.py
```

> ⚠️ **Context uzunluğu**: Ollama varsayılan olarak küçük bir context kullanıyor ve ajanlar bunu çabuk doldurur.
> Ollama'yı şöyle başlat: `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`

## Kullanım

```bash
nanocoder                          # interaktif mod
nanocoder "testleri çalıştır ve kırık olanı düzelt"   # tek seferlik
nanocoder -m qwen3:8b              # başka model
nanocoder --yolo                   # hiçbir şey sormaz, cesursan
nanocoder --base-url http://localhost:1234/v1   # LM Studio
```

| Ortam değişkeni       | Varsayılan                   |
|-----------------------|------------------------------|
| `NANOCODER_MODEL`     | `qwen2.5-coder:7b`           |
| `NANOCODER_BASE_URL`  | `http://localhost:11434/v1`  |
| `NANOCODER_API_KEY`   | boş                          |
| `NANOCODER_CTX`       | `32768`                      |

### Komutlar

| Komut | Ne yapar |
|---|---|
| `/help` | yardım |
| `/clear` | konuşmayı sıfırlar |
| `/compact` | konuşmayı özetleyip context'i boşaltır |
| `/init` | projeyi inceleyip `NANOCODER.md` yazar |
| `/model [isim]`, `/models` | model gösterir, değiştirir, listeler |
| `/yolo` | otomatik onayı açar/kapatır |
| `/think` | modelin düşüncelerini gösterir/gizler |
| `/tokens` | context kullanımını gösterir |
| `!ls -la` | shell komutu çalıştırır, çıktısı bir sonraki mesajla modele gider |
| `@src/main.py` | dosyayı mesaja ekler |

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

nanocoder -w small=qwen3:0.6b -w medium=qwen3:1.7b          # kendi ekibini kur
nanocoder -w big=llama3:8b@http://localhost:1234/v1          # başka sunucudaki model
nanocoder --no-workers                                       # tek ajan
```

- Sunucuda olmayan işçi modelleri başlangıçta otomatik devre dışı kalır
- Alt ajanların düzenleme ve komutları da aynı onay ekranından geçer. `n` dersen hem işçi hem şef durur
- Alt ajanlar başka ajan çağıramaz, yani sonsuz delegasyon zinciri yok
- `/team` ekibi gösterir. Ortam değişkeni: `NANOCODER_WORKERS="small=...,medium=..."`

## Model önerileri

Tool calling'i iyi olan modeller daha iyi sonuç verir:

- `qwen2.5-coder:7b` / `14b`: hızlı, düzgün
- `qwen3:8b`, `qwen3-coder:30b`: daha akıllı, daha aç
- 3B altı modeller de çalışır ama sonuç tam bir piyango

## Lisans

MIT
