**1. Overfitting**

Issue → model memorizes training data

Reason → small or low-diversity dataset

Solution → use **larger datasets, regularization, validation monitoring**

**2. Noisy preference data**

Issue → model learns wrong behavior

Reason → inconsistent human labels or bad synthetic data

Solution → **data filtering, multiple annotators, quality checks**

**3. Catastrophic forgetting**

Issue → model loses base capabilities after fine-tuning

Reason → large updates to weights

Solution → **small learning rate, LoRA/QLoRA, mix base-task data**

**4. Reward hacking (RLHF)**

Issue → model exploits reward model instead of improving quality

Reason → reward model imperfect or biased

Solution → **better reward model training, adversarial evaluation**

**5. Mode collapse / generic answers**

Issue → model always produces safe but boring responses

Reason → strong alignment penalties

Solution → **diverse training data, balanced preference pairs**

**6. Training instability (PPO)**

Issue → model diverges or unstable learning

Reason → large policy updates

Solution → **clipped objectives (PPO), KL penalty, smaller steps**

**7. High compute cost**

Issue → RLHF training is expensive

Reason → large models and many iterations

Solution → **LoRA/QLoRA, smaller models, distributed training**
