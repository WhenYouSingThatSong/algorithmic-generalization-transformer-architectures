### 

Self-correction: 

* Self-correct Final answer vs intermediate values.   
* Whether a mechanism is real self-correction.   
  * Wrong answer / Wrong intermediate \=\> you did not finish the compute. 3-hop \=\> 2-hop. (1-week)   
  * Mechanism 1: compute for wrong answer \=\> realize this wrong answer \=\> correction.   
  * Mechanism 2: wrong answer \=\> compute \=\> correction.   
    * You did not finish compute (1-case)   
    * Guess wrong, guess again \=\> correct answer.   
  * Mechanism 1 vs mechanism 2\. \[Scientific questions\]   
  * Realization some went wrong \=\> human concept   
    * When something wrong \=\> Is there some interesting/abnormal behavior. 

### [Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning](https://arxiv.org/abs/2510.14095?utm_source=chatgpt.com)

* Reasoning Tasks:   
  * Modular arithmetic on directed acyclic computation graphs (DAGs). ![][image1]  
  * The model processes the graph’s entire token sequence during each loop, rather than generating one reasoning token at a time. Multiple ready tokens can therefore be evaluated in parallel.   
* Intermediate reasoning representation:   
  * ${z}_{i}^{(t)}=(syntax,variable,operation,value).$  
  * A recurrent step has three main operations:   
    * H^(t+1) \= F(H^t)   
    * Z^(t+1) \= Discretize(H(t+1)   
    * H^(t+1) \= Embed (Z^t+1))   
  * The models have four different learned embedding tables: E\_s, E\_v, E-o, and E\_a.   
  * Mathematically h\_{x\_42} \= E\_s(variable) \+ E\_v(x\_42)  E\_o(N/A) \+ E\_a(empty).   
  * After a transformer loop, it uses four separate linear classification heads. Each classifier consists of learned matrix W\_f followed by argmax.   
* Loss: the model learn the correct intermediate states:   
  * L\_{align} \= \\sum\_{t=1}^T \\sum(i:di\<=t) CE(W\_vH\_i^T, vi)   
* How is the Self-correction actually implemented?   
  * Train on deliberately incorrect intermediate states, while continuing to demand correct intermediate outputs.   
    * Corrupt the value components of a random subset of discrete states during training.   
    * Trains the recurrent block to predict clean successor states.   
      * The repair supervision uses teacher-forced intermediate states.   
  * Ordinary training: F(correct\_state) \-\> correct next step.   
  * Corruption-aware training: F(incorrect state) \-\> correct next step. 

  ![][image2]

* Ablation Study:   
  * The author report that adding correction training produced little improvement for two-layer block, but substantially improved generalization with a four-layer block.   
    * Possible interpretation is that one recurrent iteration must perform multiple operations:   
      * Retrieve \-\> Recompute \-\> Continue.   
* **Existing Mechanistic Study:**   
  * Detailed Mechanistic Study on **models without self-correction**. **This is a research opportunity for us.**   
  * Most interesting questions: **whether self-correction requires a dedicated mechanism or can arise from the model repeatedly applying its ordinary reasoning algorithm.**   
* Paper was updated to 58-pages in Oct 2026\.   
  * the model learns from its own intermediate prediction errors.   
  * Two training passes:   
    * z(hat)^{t+1} \= step(z^t)   
    * Step(stopgrad(z(hat)^(t+1)) \-\> z^(t+2)   
    * The first pass predicts thenext state from a correct intermediate states. The second pass feeds the model’s prediction back into the recurrent block and trains it to prodcue the correct state two steps ahead.   
    * If the first prediction contains an error, the second pass encourages recovery.   
  * **Mechanistic novelty still opens:**   
    * The current paper still does not study the mechanism of the recovery.   
* Mechanism Hypothesis:   
  * A: Incorrect intermediate states \-\> Detect Inconsistency \-\> Recompute the affected state \-\> Resume correct computation.   
  * B: intermediate states, possibly incorrect \-\> Recompute previously solvd nodes regardless of correctness \-\> Incorrect values get overwritten autoamtiacally 

If we use this paper as starting points: 

* Mechanism of the training without self-correction \[Paper already done\]   
* Mechanism of the training with self-correction (correction in the vocabulary space)   
  * Code release.   
* Mechanism of the training with self-correction (correction in the hidden space).   
  * Self-reproduce.   
* Algorithm tasks:   
  * Ground-truth intermediate values. 

### [**Emergent Search and Backtracking in Latent Reasoning Models**](https://arxiv.org/abs/2602.08100?utm_source=chatgpt.com) 

* 2026 Feb.   
* Training the model directly, self-correction loss \=\> the author finds that the model can do self-correction.   
  * 3B-4B.   
* Central question: When a looped Transformer performs additional reasoning iterations, does it merely becomes increasingly confident in its initial answer, or can it explore alternatives, provisionally select an answer, and later reject it?  
  * Central findings: model’s decoded answer preferences often follow three stages:   
    * Exploration \=\> Commitment \=\> Revision   
    * 32% of base-condition instances exhibits a defined backtracking event.   
    * Backtracking instances have 4% higher accuracy than non-backtracking instances, an observational association.   
    * In 72% of backtracking events, the abandoned answer is the distractor most similar to the question.   
*  **Very important Research Opportunity:** The answers demonstrate reversals of decoded answer preferences. They do not casually demonstrate that the model detects and repairs an incorrect intermediate reasoning state. 

![][image3]

* Backtracking definition: A backtracking event requires an initial answer $a$ to remain the most likely answer for at least three consecutive iterations, followed by a different answer $b$ for at least three consecutive iterations, with $b$ also being the final answer. 

Research Opportunity here: **The paper** 

| Claim | Assessment |
| ----- | ----- |
| The model's preferred answer can change across recurrent iterations | Directly observed |
| More difficult distractors produce longer periods of changing predictions | Supported by controlled input comparisons |
| The model often switches away from semantically similar distractors | Supported by the reported analysis |
| The model explicitly detects that its previous answer is wrong | Not directly demonstrated |
| The model revisits and repairs an incorrect intermediate reasoning state | Not established |
| A particular attention head or circuit causes correction | Not investigated |

Comparison with the earlier work: 

|  | Recursive Latent Space Reasoning (Altabaa et al.) | Emergent Search and Backtracking (Cui & Ye) |
| ----- | ----- | ----- |
| Model | Small recurrent Transformer with discrete latent states | Pretrained Huginn 3.5B |
| Task | Modular arithmetic computation graphs | Multiple-choice QA |
| Intermediate supervision | Explicit node-value targets | No added intermediate-state supervision in this study |
| Self-correction | Trained to recover from its own intermediate predictions | Observed changes in answer preferences |
| Main evidence | Algorithmic accuracy and robustness to recurrent errors | Per-loop answer-probability trajectories |
| Detailed causal mechanism of repair | Not established | Not established |

Self-correction ability. 

Some phenomena? 

Condition 1: The model realizes there is a mistake then make the correction. 

Condition 2: The model did not realize there is a mistake just recompute the correct step. 

Condition 3: The model never corrects it. 

Hypothesis? 

1, Which Hypothesis is correct? 

2, Which training scheme will lead to which conditional? 

[image1]: ../assets/image1.png

[image2]: ../assets/image2.png

[image3]: ../assets/image3.png
