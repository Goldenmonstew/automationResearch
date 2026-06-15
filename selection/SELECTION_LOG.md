# Idea 选择记录(多模型 ensemble)

- 池子:76 idea  ·  评委:deepseek-v3.2, gpt-4o, kimi-k2.5  ·  去重后 56 unique  ·  选 top 40

| 排名 | idx | title | 总分 | manif | feas | novel | icbinb | 评委数 | 合并 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 28 | The Regularization Interference Effect: When Combi | 18.01 | 4.67 | 4.67 | 3.67 | 5.0 | 3 | [28] |
| 2 | 73 | The Brittleness Signature: Simple Diagnostic Tests | 18.0 | 4.33 | 4.67 | 4.0 | 5.0 | 3 | [73] |
| 3 | 37 | Noise as a Litmus Test: Using Controlled Label Noi | 17.68 | 4.67 | 4.67 | 3.67 | 4.67 | 3 | [37] |
| 4 | 33 | The Memorization Test: Can Simple Synthetic Tasks  | 17.67 | 5.0 | 4.67 | 3.67 | 4.33 | 3 | [33] |
| 5 | 3 | The Augmentation Paradox: When Simple Data Augment | 17.34 | 4.33 | 4.67 | 3.67 | 4.67 | 3 | [3, 4, 6] |
| 6 | 8 | Reinforce or Diversify? Selective Sample Duplicati | 17.34 | 4.0 | 4.67 | 4.0 | 4.67 | 3 | [8] |
| 7 | 20 | The Calibration Paradox: Why Data Augmentation Wor | 17.34 | 4.33 | 4.67 | 3.67 | 4.67 | 3 | [20] |
| 8 | 1 | The Informative Noise: Can High-Loss Samples Under | 17.0 | 4.0 | 4.0 | 4.0 | 5.0 | 3 | [1] |
| 9 | 69 | Confidence-Maximizing Augmentation: A Simple Infer | 17.0 | 4.33 | 4.67 | 3.67 | 4.33 | 3 | [69] |
| 10 | 13 | The Compression Test: Does Representation Compress | 16.67 | 4.0 | 4.0 | 4.67 | 4.0 | 3 | [13] |
| 11 | 25 | The Forgotten Knob: Tuning BatchNorm Gamma Initial | 16.67 | 4.33 | 5.0 | 3.67 | 3.67 | 3 | [25] |
| 12 | 40 | The First Step Matters Most: How Initial Weight Sc | 16.67 | 4.33 | 5.0 | 3.67 | 3.67 | 3 | [40, 41] |
| 13 | 60 | The Test-Time Simplicity Gap: How Small-Dataset Tr | 16.67 | 3.67 | 4.0 | 4.0 | 5.0 | 3 | [60] |
| 14 | 65 | The Learning Order Imbalance: How Uneven Class Lea | 16.67 | 4.0 | 4.33 | 3.67 | 4.67 | 3 | [65] |
| 15 | 14 | Stop When It Compresses: Representation Compressib | 16.66 | 4.0 | 4.0 | 4.33 | 4.33 | 3 | [14, 15, 21] |
| 16 | 63 | The Augmentation Pathway: Does Consistent Ordering | 16.66 | 4.33 | 4.67 | 3.33 | 4.33 | 3 | [63] |
| 17 | 22 | The Augmentation Pathway: Does Consistent Ordering | 16.34 | 4.0 | 4.67 | 3.67 | 4.0 | 3 | [22] |
| 18 | 5 | The Training Oracle: Can Simple Training Statistic | 16.33 | 4.0 | 4.0 | 4.0 | 4.33 | 3 | [5] |
| 19 | 17 | The Calibration Shake: Can Small Intentional Oscil | 16.33 | 4.0 | 4.33 | 4.0 | 4.0 | 3 | [17, 19] |
| 20 | 35 | The Right Momentum: Adaptive Batch Normalization M | 16.33 | 4.33 | 5.0 | 3.33 | 3.67 | 3 | [35] |
| 21 | 2 | The Warmup Dilemma: Does Early Weight Decay During | 16.0 | 4.0 | 4.67 | 3.33 | 4.0 | 3 | [2] |
| 22 | 10 | The Learning Rate Plateau: Does Stalling Learning  | 16.0 | 4.0 | 4.67 | 3.33 | 4.0 | 3 | [10, 11, 12] |
| 23 | 27 | Consistency Matters: Why Fixed-Order Data Augmenta | 16.0 | 4.0 | 5.0 | 3.33 | 3.67 | 3 | [27, 31] |
| 24 | 52 | The Receptive Field Timeline: Does Placing Larger  | 16.0 | 4.0 | 4.67 | 3.33 | 4.0 | 3 | [52] |
| 25 | 75 | Cleaning the Noise: Selective Suppression of Harmf | 16.0 | 4.0 | 4.67 | 3.33 | 4.0 | 3 | [75] |
| 26 | 55 | Schedule the Noise: Why Data Augmentation Strength | 15.99 | 4.33 | 4.33 | 3.33 | 4.0 | 3 | [55, 57] |
| 27 | 51 | The Order Within: How Staging Regularization Withi | 15.67 | 3.67 | 4.33 | 3.67 | 4.0 | 3 | [51] |
| 28 | 23 | One Hyperparameter to Rule Them All? A Scaling Law | 15.66 | 4.33 | 3.33 | 4.0 | 4.0 | 3 | [23] |
| 29 | 43 | Progressive Augmentation Scheduling: Growing Data  | 15.66 | 4.33 | 5.0 | 3.0 | 3.33 | 3 | [43, 44, 45, 47] |
| 30 | 67 | Clipping the Outliers: Gradient Clipping as Unexpe | 15.66 | 4.0 | 4.0 | 3.33 | 4.33 | 3 | [67, 68] |
| 31 | 16 | The Challenge Batch: Does Grouping Similar Samples | 15.34 | 3.67 | 3.67 | 4.0 | 4.0 | 3 | [16] |
| 32 | 29 | Before or After? How the Order of Operations Withi | 15.33 | 4.0 | 4.67 | 3.33 | 3.33 | 3 | [29, 30] |
| 33 | 56 | The Order of Operations: Does Temporal Sequencing  | 15.33 | 3.67 | 4.33 | 3.33 | 4.0 | 3 | [56, 70] |
| 34 | 62 | Breaking the Symmetry: Decoupling Sample Selection | 15.33 | 3.33 | 3.67 | 4.0 | 4.33 | 3 | [62] |
| 35 | 38 | Dense or Sparse? How Skip Connection Density Shape | 15.32 | 4.33 | 4.33 | 3.33 | 3.33 | 3 | [38] |
| 36 | 46 | The Early Bird Gets the Feature: How Architectural | 15.01 | 4.0 | 3.67 | 3.67 | 3.67 | 3 | [46] |
| 37 | 64 | The Activation Filter: Do Smoother Activations Act | 15.01 | 4.0 | 4.67 | 2.67 | 3.67 | 3 | [64, 66] |
| 38 | 36 | Time-Dependent Optimizer Scheduling: Dynamically R | 15.0 | 3.33 | 4.33 | 3.67 | 3.67 | 3 | [36] |
| 39 | 42 | The Small-Batch Advantage: How Batch Size Serves a | 15.0 | 4.67 | 4.67 | 2.33 | 3.33 | 3 | [42] |
| 40 | 71 | The Thinking Gap: Inserting a Computational Pause  | 15.0 | 3.0 | 3.67 | 4.0 | 4.33 | 3 | [71] |

## 各评委理由(留痕)

### [28] The Regularization Interference Effect: When Combined Regularizers Hurt More Than Help on Small Datasets
- **deepseek-v3.2**: Directly tests interference between regularizers, fitting ICBINB theme.
- **gpt-4o**: Investigates a potential failure mode of combining regularizers on small datasets.
- **kimi-k2.5**: Explicitly tests regularization failure modes through combinatorial interference on small data.

### [73] The Brittleness Signature: Simple Diagnostic Tests for Evaluating Small-Dataset Models
- **deepseek-v3.2**: Excellent ICBINB fit: diagnostic tests for brittleness are clear and fast to run.
- **gpt-4o**: Brittleness signatures are novel, feasible, and fit failure mode diagnostics.
- **kimi-k2.5**: Brittleness diagnostics are lightweight evaluations perfect for failure mode analysis.

### [37] Noise as a Litmus Test: Using Controlled Label Noise to Predict Generalization on Small Datasets
- **deepseek-v3.2**: Label noise as a diagnostic directly fits ICBINB and is highly feasible.
- **gpt-4o**: Uses label noise as a diagnostic for generalization, fitting the negative results theme.
- **kimi-k2.5**: Multiple retraining runs for noise sensitivity may approach time limit but feasible with reduced epochs.

### [33] The Memorization Test: Can Simple Synthetic Tasks Distinguish Learning from Memorization on Small Datasets?
- **deepseek-v3.2**: Clear synthetic tasks to diagnose memorization vs. learning.
- **gpt-4o**: Tests a creative diagnostic for distinguishing learning from memorization.
- **kimi-k2.5**: Synthetic tasks may not generalize to real datasets but provide clear diagnostic signals within constraints.

### [3] The Augmentation Paradox: When Simple Data Augmentations Amplify Label Noise Harm on Small Datasets
- **deepseek-v3.2**: Clear paradox on small noisy datasets, highly feasible.
- **gpt-4o**: Examines augmentation effects under label noise, fitting the theme and feasible on CIFAR-10.
- **kimi-k2.5**: Examines augmentation harm under label noise with straightforward accuracy metrics on CIFAR.

### [8] Reinforce or Diversify? Selective Sample Duplication Outperforms Standard Augmentation Under Label Noise
- **deepseek-v3.2**: Novel intervention contrasting duplication vs. augmentation.
- **gpt-4o**: Selective duplication under label noise is novel and fits the theme well.
- **kimi-k2.5**: Novel comparison of duplication versus augmentation strategies under label noise.

### [20] The Calibration Paradox: Why Data Augmentation Worsens Calibration Under Label Noise
- **deepseek-v3.2**: Extends idea 3's augmentation-noise paradox to calibration.
- **gpt-4o**: Examines augmentation effects on calibration under label noise, fitting the theme well.
- **kimi-k2.5**: Focuses on calibration degradation rather than accuracy, distinct from other augmentation-noise studies.

### [1] The Informative Noise: Can High-Loss Samples Under Label Noise Be Crucial for Distribution Shift Robustness?
- **deepseek-v3.2**: Directly tests a counterintuitive failure mode on small datasets.
- **gpt-4o**: Focuses on label noise and distribution shift, fitting the theme and feasible on CIFAR-10.
- **kimi-k2.5**: Challenges high-loss sample discarding under distribution shift, fitting ICBINB failure mode theme well.

### [69] Confidence-Maximizing Augmentation: A Simple Inference-Time Trick for Small Datasets
- **deepseek-v3.2**: Simple inference-time trick, easy to test with minimal compute.
- **gpt-4o**: Confidence-maximizing augmentation is novel, feasible, and fits failure mode exploration.
- **kimi-k2.5**: Max-confidence TTA is simple inference-time test likely to show failure.

### [13] The Compression Test: Does Representation Compressibility Predict Generalization During Training?
- **deepseek-v3.2**: Novel compressibility metric, though storage/compression adds overhead.
- **gpt-4o**: Proposes representation compressibility as a generalization metric, novel and feasible.
- **kimi-k2.5**: Novel application of compression metrics to representation learning with moderate computational overhead.

### [25] The Forgotten Knob: Tuning BatchNorm Gamma Initialization to Modulate Weight Decay's Channel-Wise Regularization
- **deepseek-v3.2**: Novel investigation of gamma init's interaction with weight decay on small datasets.
- **gpt-4o**: Explores a rarely studied interaction between BatchNorm initialization and weight decay on small datasets.
- **kimi-k2.5**: Clean hyperparameter ablation on BatchNorm gamma with measurable effects on CIFAR-10 within compute budget.

### [40] The First Step Matters Most: How Initial Weight Scale Sculpts Optimization Trajectories and Generalization on Small Datasets
- **deepseek-v3.2**: Investigates initialization scale on small datasets.
- **gpt-4o**: Examines the underexplored role of initialization scale on optimization trajectories.
- **kimi-k2.5**: Initialization scale ablation with trajectory tracking is lightweight and tests default He assumptions.

### [60] The Test-Time Simplicity Gap: How Small-Dataset Training Leads to Overconfident Predictions on Simple Test Samples
- **deepseek-v3.2**: Interesting failure mode hypothesis, but defining 'simplicity' robustly is challenging.
- **gpt-4o**: Explores confidence biases in small datasets, novel and fits failure mode theme.
- **kimi-k2.5**: Test-time analysis requires only inference and fits ICBINB failure-mode theme.

### [65] The Learning Order Imbalance: How Uneven Class Learning Dynamics Harm Generalization on Small Datasets
- **deepseek-v3.2**: Clear hypothesis about class learning imbalance, a good failure mode to study.
- **gpt-4o**: Class learning imbalance is novel, feasible, and fits failure mode exploration.
- **kimi-k2.5**: Per-class tracking is easy and imbalance diagnostics fit negative results.

### [14] Stop When It Compresses: Representation Compressibility as an Early Stopping Criterion for Small Datasets
- **deepseek-v3.2**: Applies compressibility from idea 13 to early stopping.
- **gpt-4o**: Uses compressibility for early stopping, fitting the theme and feasible.
- **kimi-k2.5**: Proposes gzip compression ratio for early stopping, challenging standard validation-based approaches.

### [63] The Augmentation Pathway: Does Consistent Ordering of Data Augmentations Improve Generalization on Small Datasets?
- **deepseek-v3.2**: Simple, clean test of augmentation ordering consistency on CIFAR-10.
- **gpt-4o**: Consistent augmentation ordering is novel, feasible, and fits failure mode exploration.
- **kimi-k2.5**: Fixed vs random augmentation order is trivial to test and likely a null result.

### [22] The Augmentation Pathway: Does Consistent Ordering of Data Augmentations Improve Generalization on Small Datasets?
- **deepseek-v3.2**: Tests a subtle, often overlooked aspect of augmentation pipelines.
- **gpt-4o**: Investigates augmentation order effects, novel and feasible on CIFAR-10.
- **kimi-k2.5**: Tests fixed versus random augmentation order with minimal implementation complexity.

### [5] The Training Oracle: Can Simple Training Statistics Outperform Validation Loss for Early Stopping on Small Datasets?
- **deepseek-v3.2**: Tests a practical failure mode of validation on small data.
- **gpt-4o**: Proposes training statistics for early stopping, feasible and fitting the theme.
- **kimi-k2.5**: Proposes training statistics over validation loss for early stopping, highly relevant to small data failure modes.

### [17] The Calibration Shake: Can Small Intentional Oscillations in Optimization Improve Model Calibration?
- **deepseek-v3.2**: Simple oscillatory perturbation to test calibration improvement.
- **gpt-4o**: Explores oscillatory SGD effects, novel and fitting the theme.
- **kimi-k2.5**: Simple modification to SGD with alternating noise for calibration improvement.

### [35] The Right Momentum: Adaptive Batch Normalization Momentum for Small Datasets
- **deepseek-v3.2**: Adaptive BN momentum for small datasets is feasible and testable.
- **gpt-4o**: Tests a straightforward modification to BatchNorm momentum for small datasets.
- **kimi-k2.5**: Adaptive scheduling of BN momentum tests whether fixed defaults are suboptimal for limited data.

### [2] The Warmup Dilemma: Does Early Weight Decay During Learning Rate Warmup Harm Generalization on Small Datasets?
- **deepseek-v3.2**: Simple, feasible intervention on a common hyperparameter practice.
- **gpt-4o**: Explores weight decay during warmup, feasible on small datasets and novel.
- **kimi-k2.5**: Tests WD scheduling during warmup with simple scaling interventions on standard small datasets.

### [10] The Learning Rate Plateau: Does Stalling Learning Rate Decay After Warmup Improve Generalization on Small Datasets?
- **deepseek-v3.2**: Simple, feasible LR schedule modification for small datasets.
- **gpt-4o**: Explores learning rate plateau effects, feasible and fitting the theme.
- **kimi-k2.5**: Tests constant LR plateau duration with simple schedule modifications.

### [27] Consistency Matters: Why Fixed-Order Data Augmentation Improves Generalization on Small Datasets
- **deepseek-v3.2**: Tests fixed vs. random augmentation order on small datasets.
- **gpt-4o**: Tests a plausible hypothesis about augmentation order effects on small datasets.
- **kimi-k2.5**: Simple augmentation pipeline manipulation testing whether randomization actually helps empirical performance.

### [52] The Receptive Field Timeline: Does Placing Larger Kernels Early Improve Generalization on Small Datasets?
- **deepseek-v3.2**: Simple, controlled CNN variants can clearly test kernel placement on CIFAR-10.
- **gpt-4o**: Investigates kernel size ordering, a novel hypothesis with clear small-dataset applicability.
- **kimi-k2.5**: Kernel ordering is easy to test and likely to show null or negative results.

### [75] Cleaning the Noise: Selective Suppression of Harmful Gradient Noise Components for Small-Dataset Generalization
- **deepseek-v3.2**: The idea is a clear, testable modification of SGD that could show a measurable effect on CIFAR-10 subsets within the time and compute constraints.
- **gpt-4o**: The idea is feasible within the constraints, novel in its approach to gradient noise decomposition, and fits the theme by exploring potential failure modes in optimization.
- **kimi-k2.5**: ResNet-18 on CIFAR-10 subsets is feasible within one hour and the generalization effect is measurable without saturation, though the gradient decomposition approach parallels existing variance-reduction methods and likely fails to show consistent improvements on small datasets, fitting the negative results theme.

### [55] Schedule the Noise: Why Data Augmentation Strength Should Be Scheduled Like Learning Rate for Small Datasets
- **deepseek-v3.2**: Clear hypothesis about augmentation scheduling that is testable on CIFAR-10.
- **gpt-4o**: Scheduling augmentation strength is novel, feasible, and fits failure mode exploration.
- **kimi-k2.5**: Augmentation scheduling is feasible though curriculum augment is known.

### [51] The Order Within: How Staging Regularization Within a Training Batch Improves Small-Dataset Generalization
- **deepseek-v3.2**: Tests a specific, implementable hypothesis about regularization staging within a batch.
- **gpt-4o**: Novel staged regularization idea, feasible on small datasets, fits failure mode exploration.
- **kimi-k2.5**: Intra-batch regularization staging is feasible and novel but success is uncertain.

### [23] One Hyperparameter to Rule Them All? A Scaling Law for Weight Decay on Small Datasets
- **deepseek-v3.2**: Large hyperparameter sweep but highly relevant to small datasets.
- **gpt-4o**: Proposes a scaling law for weight decay, novel and fitting the theme.
- **kimi-k2.5**: Requires 270+ full training runs, far exceeding 1-hour compute constraint.

### [43] Progressive Augmentation Scheduling: Growing Data Distortion as Training Progresses for Better Small-Dataset Generalization
- **deepseek-v3.2**: Progressive augmentation scheduling for small datasets.
- **gpt-4o**: Explores progressive augmentation scheduling, a plausible but less novel idea.
- **kimi-k2.5**: Curriculum learning applied to augmentation strength tests whether constant distortion is suboptimal.

### [67] Clipping the Outliers: Gradient Clipping as Unexpected Regularization on Small Datasets
- **deepseek-v3.2**: Simple gradient clipping sweep, easy to implement and test on CIFAR-10.
- **gpt-4o**: Gradient clipping as regularization is novel, feasible, and fits failure mode exploration.
- **kimi-k2.5**: Threshold sweep requires multiple runs pushing 1-hour limit.

### [16] The Challenge Batch: Does Grouping Similar Samples Improve Generalization on Small Datasets?
- **deepseek-v3.2**: Requires pretrained features or clustering, adding complexity.
- **gpt-4o**: Investigates batching strategies, novel and feasible on CIFAR-10.
- **kimi-k2.5**: Similarity batching requires per-epoch clustering, adding computational overhead but feasible on CIFAR-10.

### [29] Before or After? How the Order of Operations Within Optimizer Steps Affects Generalization on Small Datasets
- **deepseek-v3.2**: Investigates optimizer operation order on small datasets.
- **gpt-4o**: Examines an underexplored aspect of optimizer step ordering on small datasets.
- **kimi-k2.5**: Low-level optimizer modification testing weight decoupling order with likely minimal practical difference.

### [56] The Order of Operations: Does Temporal Sequencing of Regularization Techniques Improve Small-Dataset Generalization?
- **deepseek-v3.2**: Tests a specific temporal ordering of regularizers, feasible with ResNet-18.
- **gpt-4o**: Overlaps with idea 51 but focuses on temporal sequencing rather than within-batch staging.
- **kimi-k2.5**: Phased regularization is trivial to implement and test.

### [62] Breaking the Symmetry: Decoupling Sample Selection from Parameter Update with Dual Loss Functions for Small Datasets
- **deepseek-v3.2**: Complex training loop modification but tests an interesting decoupling hypothesis.
- **gpt-4o**: Decoupling loss functions for sample selection and updates is novel and fits failure mode theme.
- **kimi-k2.5**: Dual loss mechanism is implementable but manifestability depends on small data overfitting.

### [38] Dense or Sparse? How Skip Connection Density Shapes Small-Dataset Learning Dynamics
- **deepseek-v3.2**: Architectural modification of skip connection density.
- **gpt-4o**: Tests the impact of skip connection density on small-dataset generalization.
- **kimi-k2.5**: Architecture modification requiring multiple full training runs of different ResNet variants.

### [46] The Early Bird Gets the Feature: How Architectural Inductive Biases Shape Early Learning Dynamics on Small Datasets
- **deepseek-v3.2**: Architectural learning dynamics requires multiple models and detailed tracking.
- **gpt-4o**: Investigates how architectural biases shape early learning dynamics on small datasets.
- **kimi-k2.5**: CKA similarity computation and per-class tracking across three architectures likely exceeds time budget.

### [64] The Activation Filter: Do Smoother Activations Act as Implicit Regularizers on Small Datasets?
- **deepseek-v3.2**: Straightforward drop-in activation function test on small CIFAR subsets.
- **gpt-4o**: Smooth activation functions as regularizers is feasible and fits small-dataset generalization.
- **kimi-k2.5**: Activation function comparison is well-trodden and likely shows minor differences.

### [36] Time-Dependent Optimizer Scheduling: Dynamically Reordering Operations for Better Small-Dataset Generalization
- **deepseek-v3.2**: Extends optimizer order idea with dynamic scheduling.
- **gpt-4o**: Investigates dynamic optimizer operation scheduling, a novel but complex idea.
- **kimi-k2.5**: Dynamic reordering of optimizer operations extends static ablations to time-dependent failure analysis.

### [42] The Small-Batch Advantage: How Batch Size Serves as an Implicit Regularizer on Small Datasets
- **deepseek-v3.2**: Well-known batch size effect, but clean test on small datasets.
- **gpt-4o**: Tests a well-known hypothesis about small batch sizes as implicit regularizers.
- **kimi-k2.5**: Batch size regularization is well-studied but confirms negative results for large batches on small data.

### [71] The Thinking Gap: Inserting a Computational Pause Between Forward and Backward Passes for Better Small-Dataset Learning
- **deepseek-v3.2**: Interesting 'thinking gap' hypothesis, but implementation adds training complexity.
- **gpt-4o**: Inserting a computational pause is novel but may have limited manifestability on small datasets.
- **kimi-k2.5**: Forward-backward intervening step is easy to add and likely provides no benefit.

