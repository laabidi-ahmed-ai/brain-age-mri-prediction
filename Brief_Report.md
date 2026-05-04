# Brain Age Prediction from MRI using Deep Learning

---

## 1. Engage Phase Summary

Chronological age is reflected in measurable structural changes in the brain. **Brain age prediction** from magnetic resonance imaging (MRI) estimates a subject’s age directly from neuroanatomy, producing a single interpretable summary of complex spatial patterns. When the gap between predicted brain age and true age is studied systematically, it can support research into accelerated aging and neurodegenerative conditions; therefore, accurate and reliable models are clinically and scientifically relevant.

The **problem** addressed in this project is **regression**: given MRI data, predict **age in years**. The **objective** is to build a deep learning pipeline that generalizes well on held-out subjects, using the **OASIS** cross-sectional dataset , a publicly available dataset of brain MRI scans from subjects of varying ages, and to justify modeling and training choices with clear experiments and validation metrics.

The **chosen approach** is **convolutional neural networks** with **transfer learning**. Three-dimensional MRI volumes are decomposed into two-dimensional axial slices for model input, which allows leveraging mature 2D image backbones and stable training procedures while remaining aligned with common practice in slice-based brain MRI analysis. The engagement phase established that deep learning was appropriate given the high dimensionality of MRI and the need to learn hierarchical spatial features without hand-crafted radiomic descriptors.

---

## 2. Investigate Phase Findings

**Notebook 1 (model selection)** implemented a deliberate **screen-then-scale** strategy. Multiple pretrained architectures—**EfficientNet-B0**, **ResNet18**, **ResNet34**, and **DenseNet121**—were compared on a **small subset** (on the order of **35–40 subjects**) to obtain a fast, reproducible ranking under constrained compute. The comparison focused on validation behavior and ranking stability rather than claiming final clinical performance from the subset alone.

**EfficientNet-B0** emerged as the **strongest candidate** under this controlled comparison, balancing predictive quality and efficiency relative to the alternatives tested. To avoid premature investment in weaker architectures, the investigation then **scaled training** to a **larger multi-disc configuration** (**12 discs**, **15 epochs** for the scaled stage). On the test split, the scaled selection workflow achieved a **test MAE of approximately 4.59 years** (subject-level error reported in line with the project’s primary metric).

**Notebook 2 (refinement)** held the backbone fixed at EfficientNet-B0 and shifted the scientific question to **training strategy and regularization**. **Phase 1** compared four variants: a **baseline** fine-tuning setup (**learning rate 1e-4**), a **lower learning rate** (**1e-5**), **dropout 0.3**, and a **frozen backbone** (head-only training). The **frozen backbone** configuration performed poorly, indicating that **full fine-tuning** was necessary for this MRI distribution. The **dropout** configuration was selected for **Phase 2** despite a small numeric gap versus the baseline, prioritizing **generalization** under limited medical imaging data.

**Phase 2** performed **controlled hyperparameter tuning** on **dropout (0.2 vs 0.3)** and **learning rate (1e-4 vs 5e-5)**. The best configuration was **dropout 0.3 with learning rate 5e-5**, reaching a **final MAE of approximately 4.3391 years** on the validation-oriented evaluation used in the notebook workflow. Together, the investigation showed that (i) backbone choice matters and EfficientNet-B0 was empirically preferred, (ii) freezing pretrained features was inadequate, and (iii) modest regularization combined with a tuned learning rate improved the reported generalization metric.

---

## 3. Act Phase Architecture

**Data and preprocessing**

- **Input representation:** MRI **volumes** are processed into **2D axial slices** used as network inputs.
- **Preprocessing:** intensity handling and resizing consistent with the torchvision pipeline (e.g., **224×224**), with **subject-level splitting** to reduce leakage across slices from the same individual.

**Model**

- **Backbone:** **EfficientNet-B0** (ImageNet-pretrained weights), selected after comparative screening.
- **Head:** regression output predicting **age in years** (a fully connected regression head predicting age in years.).

**Training strategy**

- **Transfer learning:** pretrained initialization followed by **full fine-tuning** (not frozen-feature extraction in the final system).
- **Regularization:** **dropout** integrated in the training configuration; final selected setting **0.3**.
- **Optimization:** **SmoothL1Loss** as the training objective; learning rate schedule consistent with the tuned configuration (**5e-5** in the best run).
- **Tuning discipline:** two-phase refinement—first compare training regimes, then run a **small factorial-style comparison** on dropout and learning rate to avoid confounding.

**Evaluation**

- **Primary metric:** **Subject MAE** (aggregate predictions per subject before error computation).
- **Secondary metric:** **Slice MAE** for diagnostic visibility into slice-level behavior.

**Explainability**

- **Grad-CAM** was applied to visualize salient regions driving predictions. Qualitative inspection indicated attention concentrated on **anatomically plausible brain structures** rather than background, supporting that the learned representation aligns with meaningful neuroanatomy for the stated regression task.

---

Overall, this project demonstrates that a structured workflow combining model selection and controlled refinement can effectively improve performance in MRI-based age prediction tasks.


