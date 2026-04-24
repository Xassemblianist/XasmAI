# XasmAI

```
██╗  ██╗ █████╗ ███████╗███╗   ███╗     █████╗ ██╗
╚██╗██╔╝██╔══██╗██╔════╝████╗ ████║    ██╔══██╗██║
 ╚███╔╝ ███████║███████╗██╔████╔██║    ███████║██║
 ██╔██╗ ██╔══██║╚════██║██║╚██╔╝██║    ██╔══██║██║
██╔╝ ██╗██║  ██║███████║██║ ╚═╝ ██║    ██║  ██║██║
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝    ╚═╝  ╚═╝██║
```

**Sıfırdan, bağımlılıksız C++20 neural network + CUDA GPT eğitim motoru.**

PyTorch yok. TensorFlow yok. Sadece cuBLAS + elle yazılmış CUDA kernel'ları + modern C++.

---

## Ne var içinde

**CPU tarafı** (header-only, `include/xasmai/`):
- Matris, katmanlar, aktivasyonlar (ReLU/GELU/SiLU/Tanh/Softmax)
- Loss fonksiyonları (MSE, BCE, CE)
- Optimizer'lar (SGD, Momentum, Adam)
- LSTM, Attention, Transformer, BPE tokenizer

**GPU tarafı** (`cuda/gpt_cuda.cu`):
- Modern GPT mimarisi: **RMSNorm + RoPE + SwiGLU FFN + causal self-attention**
- **AdamW** weight decay + **cosine LR schedule** + warmup
- **Gradient clipping** (global L2 norm)
- **Top-k / top-p sampling**
- TF32 tensor core matmul'leri (cuBLAS)
- Weight tying (input/output embedding)
- Checkpoint kaydet/yükle, SIGINT ile güvenli kapanış
- Pre-allocated workspace (per-step allocation yok)

---

## Hızlı başlangıç (Windows + 5070 Ti)

**Çift tıkla çalıştır:**

```
gpt_cuda.exe
```

Interaktif wizard açılır:
1. Mod seç — yeni eğit / devam et / sadece metin üret
2. Corpus dosyanı listeden seç (`.txt` / `.md`)
3. Model boyutunu seç:

| Preset | dim | heads | layers | Parametre | Hedef VRAM |
|--------|-----|-------|--------|-----------|------------|
| Tiny   | 128 | 4     | 4      | ~1M       | <1 GB      |
| Small  | 256 | 8     | 6      | ~10M      | ~2 GB      |
| Medium | 512 | 8     | 8      | ~50M      | ~6 GB      |
| Large  | 768 | 12    | 12     | ~150M     | ~12 GB     |
| XL     | 1024| 16    | 16     | ~400M     | ~15 GB     |
| Özel   | elle girin                                        |

4. Adım sayısı + learning rate ver
5. Eğitim biter → "şimdi metin üret?" sorar → prompt yaz, devam ettirir

---

## Inference komutları

Metin üretim modunda prompt satırı:
- Normal metin → devamını üretir
- `/temp 0.5` — sampling sıcaklığı
- `/len 300` — üretim uzunluğu
- `/q` — çıkış

---

## CLI modu (script'lere için)

```bash
./gpt_cuda.exe corpus.txt \
    --dim 512 --heads 8 --layers 8 --ff 2048 --seq 256 \
    --lr 3e-4 --steps 100000 --warmup 1000 \
    --clip 1.0 --top-k 40 --top-p 0.95 \
    --ckpt my_model
```

Devam etmek için:
```bash
./gpt_cuda.exe --resume my_model_latest.bin
```

---

## Derleme

```bash
make gpt_cuda CUDA_ARCH=sm_120   # RTX 50xx (Blackwell)
make gpt_cuda CUDA_ARCH=sm_89    # RTX 40xx (Ada)
make gpt_cuda CUDA_ARCH=sm_86    # RTX 30xx (Ampere)
make gpt_cuda CUDA_ARCH=sm_75    # RTX 20xx (Turing)
```

**Gereksinimler:** CUDA Toolkit 12+, C++20 compiler, cuBLAS.

CPU-only örnekler (XOR, regresyon, RL agent, GAN, CPU GPT):

```bash
make xor regression agent gan gpt
```

---

## Mimari notlar

Bu bir "mini GPT" değil — 5070 Ti üstünde 400M param'a kadar eğitim yapabilen tam bir motor. Ama sıfırdan trilyon token pre-train etmek için değil. Gerçekçi kullanım:

- **Domain SLM** (Small Language Model): ERP, kod, log analizi, yapısal görevler için 10M-200M param özel model
- **Araştırma / öğrenme** — CUDA kernel + transformer matematiğini açık, okunabilir kodda görmek
- **Açık modele fine-tune tabanı** (henüz loader yok — yol haritasında)

---

## Yol haritası

- [ ] FP16 / BF16 mixed precision (tensor core hızlanması)
- [ ] KV-cache'li inference (uzun metin üretim hızı)
- [ ] Gradient accumulation (büyük effective batch)
- [ ] FlashAttention-tarzı fused softmax (uzun context)
- [ ] BPE tokenizer entegrasyonu (char-level yerine)
- [ ] LoRA fine-tuning
- [ ] Açık ağırlık loader (Qwen / Llama / Mistral)

---

## Lisans

MIT
