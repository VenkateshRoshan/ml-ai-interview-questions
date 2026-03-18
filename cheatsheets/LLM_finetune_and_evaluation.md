## PART 1 - TRAINING & FINE-TUNING

### The "When to Touch the Model" Hierarchy

Always start with the cheapest option and move right only when you must:

```
Prompt Engineering → RAG → Fine-tuning → Pre-training
     (free)           (cheap)  (expensive)   (very expensive)
```

**Prompt engineering** when the model already knows how, it just needs better instructions.
**RAG** when the problem is *knowledge* - facts, documents, freshness.
**Fine-tuning** when the problem is *behavior* - tone, format, domain reasoning the model consistently gets wrong.
**Both RAG + Fine-tune** when you need the model to *act differently* AND *know more*.

---

### The "How to Fine-Tune Efficiently" Hierarchy

```
Full Fine-tuning → LoRA → QLoRA
(all weights)    (1% weights)  (4-bit + LoRA)
```

- **LoRA** injects two tiny matrices alongside frozen weights. Same quality, 100x fewer parameters to train.
- **QLoRA** quantizes the base model to 4-bit first, *then* applies LoRA. Made 70B fine-tuning possible on a single GPU.
- **Rule of thumb:** Good GPU memory → LoRA. Single consumer GPU → QLoRA.

---

### The "How to Align a Model" Story

Raw pre-training gives you a text predictor. Alignment makes it *useful and safe.*

```
Pre-training → SFT → Reward Model → PPO (RLHF)
                              ↓
                            DPO (simpler, same result)
```

- **SFT** - teach format and behavior with demonstrations
- **RLHF** - humans rank outputs → reward model learns preferences → PPO optimizes the LLM against it
- **DPO** - skips the reward model entirely, directly learns from preference pairs. Simpler, stable, widely adopted now
- **Instruction tuning** - a form of SFT specifically on (instruction → response) pairs. Bridges "text predictor" and "assistant"

---

### Scalable Training - The Three Bottlenecks

As you scale, the bottleneck shifts:

1. **Compute** → Use mixed precision (bf16), Flash Attention, gradient checkpointing
2. **Memory** → Model parallelism, tensor parallelism, DeepSpeed ZeRO sharding
3. **Communication** → Pipeline parallelism, efficient gradient sync

Data side: stream don't load, deduplicate aggressively, bad data at scale is catastrophic.

---

### Inference Optimization

**Quantization** - Reduce weight precision (32-bit → 8-bit → 4-bit). Free speed gains from reduced memory bandwidth. INT8 is almost always a free win. 4-bit has slight quality loss but fits huge models.

**Speculative decoding** - Small draft model generates tokens fast → big model verifies in parallel → 2-3x speedup, zero quality loss.

**Time to First Token (TTFT)** - How long until the user sees *anything*. Determines perceived responsiveness. Optimize with smaller models, KV caching, speculative decoding.

---

### Designing a Math Model (Template for Any Domain)

This is your blueprint for any "design a model for X" question:

1. **Data** - high quality step-by-step reasoning examples (not just answers)
2. **SFT** - teach chain-of-thought format
3. **Post-training** - use a verifier as reward signal (rule-based or model-based), reinforce correct reasoning trajectories
4. **Eval** - held-out benchmarks + faithfulness of reasoning, not just final answer accuracy

---

## PART 2 - EVALUATION & MONITORING

### The Metrics Stack - Three Levels

```
Business Metrics → Product Metrics → Model Metrics
"reduce costs"  → "deflection rate" → "task completion accuracy"
```

Never report model metrics without connecting them to business outcomes. That's senior-level thinking.

**Key business metrics to always mention:**
- Deflection rate, Win rate, CSAT/NPS
- P95/P99 latency (tail latency kills UX)
- Cost per query, Time-to-resolution

---

### The Hallucination Defense Stack

This comes up in *many* different forms. One mental model covers them all:

```
PREVENT → DETECT → FILTER → MONITOR
```

- **Prevent** - RAG with vetted sources, citations, prompt constraints ("only use provided context")
- **Detect** - faithfulness scoring, consistency sampling, LLM-as-judge
- **Filter** - post-processing layer that blocks low-confidence or ungrounded responses
- **Monitor** - track hallucination rate as an ongoing production metric, sample + auto-score continuously

For **medical specifically** - add conservative refusal, human-in-the-loop for critical decisions, and red-team aggressively for medical misinformation. A confident wrong answer in medicine is worse than "I don't know."

---

### Debugging a Confidently Wrong RAG Chatbot

Walk this checklist in order:

1. **Retrieval problem?** → Wrong chunks came back, check the retrieved docs
2. **Context ignored?** → Right docs retrieved but model used parametric memory instead
3. **Chunking problem?** → Answer spans chunk boundaries, critical info was cut off
4. **Model hallucinating despite context?** → Strengthen prompt constraints
5. **No confidence signal?** → Add a faithfulness check layer before response reaches user

---

### Metrics - What They Measure & Where They Break

| Metric | Good for | Breaks when |
|--------|----------|-------------|
| Perplexity | Comparing LMs on same domain | Different tokenizers, different domains |
| BLEU | Translation quality | Paraphrases, creative outputs |
| ROUGE | Summarization | Factually wrong but n-gram similar outputs |
| BERTScore | Semantic similarity | Still doesn't catch factual errors |
| LLM-as-judge | Open-ended quality | Judge model has its own biases |

**The direction of travel:** industry is moving away from n-gram metrics toward LLM-as-judge + human preference for anything open-ended.

---

### Testing Non-Deterministic Outputs

You can't do `assert output == expected`. Instead:

- **Property tests** - test what the output *must satisfy*, not what it must *equal*
- **Consistency tests** - run same prompt 10x, measure variance
- **Semantic similarity** - embedding distance to a canonical good answer
- **Regression sets** - known past failures; if they still fail, don't ship

---

### The Golden Dataset - Your Most Important Eval Asset

- Real representative queries + high-quality human annotations
- Include edge cases and known failure modes *deliberately*
- Version it like production code
- Run every model change against it automatically before shipping
- It's your regression test suite for model behavior

---

### Calibration - The Underrated Concept

Two models with identical accuracy: **always choose the better-calibrated one.**

- **Calibration** = does the model's confidence match its actual accuracy?
- A model that's "90% confident" should be right 90% of the time
- Poorly calibrated = confidently wrong = dangerous in production
- Measure with **ECE (Expected Calibration Error)**
- Fix with **temperature scaling** (simple, effective post-hoc method)

---

### Deployment Testing - Never Go Straight to 100%

```
Shadow → Canary (1-5%) → Interleaved → Full A/B → Ramp to 100%
(no users)  (real users,     (same session,    (proper split)
             small blast)     controls variance)
```

Always have a rollback plan before you deploy anything.

---

### Diagnosing a Production Accuracy Drop

**Never retrain immediately.** Diagnose first:

1. Did the *evaluation* change? (ruler got shorter)
2. Did the *input distribution* drift? (new user behaviors)
3. Did something *upstream* change? (RAG, API, prompt template)
4. Is the drop *uniform or in a specific slice*? (tells you where to look)
5. Did the model get *silently updated* by a provider?

Fix the root cause, not the symptom. Retrain only if the problem is actually in the model.

---

### SHAP, LIME & Interpretability - One Paragraph Answer

Both answer "why did the model predict this?" LIME perturbs the input locally and fits a simple model on top. SHAP uses game theory to fairly attribute credit to each feature. For LLMs, they give *approximations* not ground truth. Use them for debugging surprising outputs and catching bias patterns, not as definitive explanations.

---

### The Feedback Flywheel - How Systems Improve

```
Users interact
      ↓
Collect signals (explicit: thumbs, implicit: edits/rejections)
      ↓
Identify failure modes
      ↓
Improve data or model
      ↓
Better model → better user experience → better signals
      ↓
              (repeat)
```

Three loops running in parallel:
- **Days** - prompt refinement
- **Weeks** - fine-tuning on curated preference data
- **Months** - training data curation improvements

---

### Bias & Fairness - The Honest Answer

Detect by measuring performance *disaggregated by demographic group*. A 90% average that hides a 70% for one group is a biased model.

The honest tradeoff: safety filters over-refuse certain groups. You tune the threshold, monitor by group, document your decisions, and accept there's no perfect setting - only a *defensible* one. That's the mature answer.

---

## THE CHEAT SHEET FOR ANY QUESTION YOU GET STUCK ON

If you blank on a question, structure your answer around:

**"It depends on three things..."**
1. The scale / constraints (compute, data, latency budget)
2. The tradeoffs involved (quality vs. cost, safety vs. helpfulness)
3. What you'd measure to know if it's working

This framework gets you through almost any LLM systems design question with confidence.

---

## LAST-MINUTE REMINDERS

- **Hallucinations** - you can reduce, not eliminate. Always multi-layered defense.
- **RAG vs Fine-tune** - RAG for knowledge, fine-tune for behavior, often both.
- **Metrics** - no single metric is enough. Always a portfolio. Always connect to business.
- **Calibration** - a confident wrong model is worse than an uncertain honest one.
- **Production monitoring** - offline eval is a snapshot. You need live monitoring.
- **Bias** - always mention disaggregated evaluation, not just averages.
- **DPO > PPO** for most practical alignment work today - simpler and stable.
- **QLoRA** - the reason fine-tuning large models became accessible to everyone.
