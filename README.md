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

- **Yerel model, en hızlı altyapıda**: makinedeki AgentCoder Server / Ollama / llama.cpp / vLLM / SGLang / LM Studio sunucularını bulur, her model için en hızlısını ölçüp seçer
- **Kendi inference sunucusu**: `agentcoder-server`, agent iş yüküne özel (prefix KV cache, CPU'da bf16 prefill + int8 decode)
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
| AgentCoder Server | 8090 |
| Ollama | 11434 |
| llama.cpp (`llama-server`) | 8080–8083 |
| vLLM | 8000 |
| SGLang | 30000 |
| LM Studio | 1234 |

İsimleri farklı da olsa (`qwen2.5-coder:7b`, `Qwen/Qwen2.5-Coder-7B-Instruct`, `qwen2.5-coder-7b-instruct-q4_k_m.gguf`)
aynı modeli tanır. Bir model birden fazla sunucudaysa **gerçek bir agent turunu** ölçer (1.2k token'lık
soğuk istek + tool sonucu eklenmiş ikinci tur) ve en hızlısını kullanır. "hi" yazıp hız ölçmek agent için yalan söyler.
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

## AgentCoder Server (kendi altyapımız)

Agent'lar her turda tüm konuşmayı baştan yollar. Genel amaçlı sunucular bunun için optimize değil.
`agentcoder_server.py` agent iş yüküne göre yazılmış küçük, OpenAI uyumlu bir sunucu:

- **Prefix KV cache**: son isteğin KV cache'ini tutar, sadece yeni eklenen kısmı hesaplar
- **CPU'da iki ağırlık seti**: prefill bf16 ile (AMX/AVX512-BF16'da hesap gücü belirleyici),
  decode per-channel int8 ile (bellek bant genişliği belirleyici, int8 yarı byte). Greedy çıktı bf16 ile birebir aynı, decode 1.7× hızlı
- **LoRA adapter'ları** yüklerken modele merge eder, Qwen `<tool_call>` çıktısını native `tool_calls`'a çevirir
- Tüm inference ana thread'de: her HTTP isteğinde yeni thread açınca OpenMP havuzu soğuk başlıyor, %30 hız kaybı

```bash
pip install "agentcoder[server] @ git+https://github.com/gubir34/NanoCoder1"   # torch, transformers, peft
agentcoder-server \
    --model qwen2.5-coder:7b=Qwen/Qwen2.5-Coder-7B-Instruct \
    --model agentcoder-worker:0.5b=Qwen/Qwen2.5-Coder-0.5B-Instruct+finetune/adapters/agentcoder-worker-0.5b
agentcoder -w small=agentcoder-worker:0.5b      # 8090'da otomatik bulunur
```

**Ölçüm** (4 çekirdekli CPU, AMX var, Qwen2.5-Coder-1.5B, `scripts/bench_backends.py`):

| Altyapı | Ağırlık | Soğuk istek: ilk token | Agent turu: ilk token | Decode |
|---|---|---|---|---|
| Ollama 0.40 | q8_0 GGUF | 4.66 s | 3.42 s | 22–24 tok/s |
| llama.cpp (`llama-server`) | q8_0 GGUF | 4.63 s | 3.74 s | 22–24 tok/s |
| **AgentCoder Server** | bf16 + int8 | **2.27 s** | **1.78 s** | 16 tok/s |

Biz prefill'de ~2× hızlıyız, decode'da llama.cpp hâlâ önde. Agent turlarında model çoğunlukla 20–60 token'lık
tool çağrısı yazdığı için toplamda biz kazanıyoruz, otomatik seçim de bu makinede AgentCoder Server'ı seçti.
Uzun cevap yazan bir sohbette llama.cpp daha iyi olur. GPU'da vLLM/SGLang'i geçmeye çalışmıyoruz.

## Fine-tune edilmiş işçiler

Ham `qwen2.5-coder:0.5b` sub-agent olarak **işe yaramıyor**: tool çağırmak yerine ```json bloğu yazıyor.
Bunun için `finetune/` altında uçtan uca bir hat var:

1. `gen_data.py`: rastgele mini Python projeleri üretir. Scripted bir "öğretmen" görevi agentcoder'ın **gerçek
   tool'larıyla** çözer, her örnek bir kontrol fonksiyonundan geçer (yanlış örnek veri setine giremez)
2. `train.py`: LoRA (r=16, tüm projeksiyonlar), sadece asistan token'larında loss, CPU'da bf16
3. `eval.py`: farklı seed'lerle üretilmiş görevleri **gerçek agent döngüsünde** çalıştırır, sonucu dosyalardan kontrol eder

Önemli bulgu: küçük Qwen modellerinde `<tool_call>` ve `</tool_call>` token'larının embedding'leri **birebir aynı**
(eğitilmemiş) ve `<|im_start|>` ile aynı yönde. LoRA bunu düzeltemiyor: loss 0.3'e iniyor ama model yine tool çağırmıyor.
Çözüm bu iki satırı ayrı, yüksek learning rate ile eğitmek (`trainable_token_indices`). Eğitim sonunda bir
"ilk token" kontrolü var, ortalama loss bir daha böyle bir hatayı saklayamaz.

**Sonuçlar** (30 görev: dosya bulma, kullanım arama, özetleme, bug düzeltme, rename, sabit değiştirme, fonksiyon ekleme):

| Model | Başarı | Tool hatası |
|---|---|---|
| `qwen2.5-coder:0.5b` (ham) | 0/30 | – (hiç tool çağıramıyor) |
| `agentcoder-worker:0.5b` | **26–28/30** (3 koşu) | 0–7 |

Okuma/arama/küçük düzeltme görevlerinde neredeyse kusursuz. Zayıf olduğu yer yeni fonksiyon yazmak
(0/2): kodu yanlış yazıp "testler geçti" diyor. 0.5B'den mucize beklemeyin, o yüzden `small` rolü zaten salt okunur.
Not: eval görevleri eğitimle aynı dağılımdan (farklı seed'ler), yani bu "agentcoder'ın tool akışını öğrendi mi"
ölçümü, genel zeka testi değil.

```bash
python finetune/gen_data.py --n 800 --seed 1 --out finetune/data/train.jsonl
python finetune/train.py --base Qwen/Qwen2.5-Coder-0.5B-Instruct --data finetune/data/train.jsonl \
    --out finetune/adapters/my-worker            # 4 çekirdek CPU'da ~1 saat
agentcoder-server --model w=Qwen/Qwen2.5-Coder-0.5B-Instruct+finetune/adapters/my-worker &
python finetune/eval.py --url http://127.0.0.1:8090/v1 --model w --n 30
python finetune/export_ollama.py --adapter finetune/adapters/my-worker   # Ollama'ya aktar (opsiyonel)
```

## Model önerileri

Tool calling'i iyi olan modeller daha iyi sonuç verir:

- `qwen2.5-coder:7b` / `14b`: hızlı, düzgün
- `qwen3:8b`, `qwen3-coder:30b`: daha akıllı, daha aç
- 3B altı ham modeller tam bir piyango. İşçi olarak fine-tune'lu `agentcoder-worker` kullanın

## Lisans

MIT
