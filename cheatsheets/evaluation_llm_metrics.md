## Depends on the **type of benchmark**, these are the common metrics used.

**1. Accuracy**
Used for reasoning benchmarks like **MMLU, GSM8K, ARC**.
Checks if the **final answer matches the ground truth**.

**2. Exact Match (EM)**
Used in **QA benchmarks (SQuAD, Natural Questions)**.
Answer must **exactly match the reference answer**.

**3. F1 Score**
Used when answers may vary slightly.
Measures **token overlap between prediction and ground truth**.

**4. Pass@k**
Used in **coding benchmarks (HumanEval)**.
Measures if **at least one of k generated programs passes all tests**.

**5. BLEU / ROUGE / BERTScore**
Used for **text generation or summarization** benchmarks.

**6. Human / LLM preference scores**
Used for **chat quality, helpfulness, safety** evaluations.

So typically **accuracy, EM/F1, pass@k, and preference scores** are the main metrics.

## In **production**, companies usually track **three types of metrics**.

### 1. Quality metrics

* **Answer accuracy / task success rate**
* **Faithfulness / hallucination rate** (especially for RAG)
* **User satisfaction / thumbs up–down**
* **LLM-as-judge scores**

### 2. Safety metrics

* **Toxicity rate**
* **PII leakage rate**
* **Policy violation rate**
* **Refusal rate / deflection rate**

### 3. System metrics

* **Latency (TTFT, total response time)**
* **Token usage / cost per request**
* **Throughput**
* **Error / fallback rate**

So in production we monitor **quality, safety, and system performance**, not just benchmark accuracy.
