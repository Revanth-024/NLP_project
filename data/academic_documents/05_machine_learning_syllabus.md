# Machine Learning Syllabus

**Synthetic educational data created for NLP workshop purposes.**

Hindu College of Engineering is a completely fictional educational institution created only for this workshop.

## Course information

| Field | Value |
| --- | --- |
| Course code | HCE-CSE502 |
| Course name | Machine Learning |
| Semester | 5 |
| Credits | 4 |
| Delivery | theory |
| Faculty | Dr. Veyana Teral |
| Academic year | 2026-2027 |

## Course description

Data-driven methods that learn patterns from examples to make predictions on unseen observations.

## Prerequisites

Earlier-course prerequisites: Probability and Statistics (HCE-CSE303); Design and Analysis of Algorithms (HCE-CSE304); Linear Algebra for Computing (HCE-CSE404).

The related course is Machine Learning Laboratory (HCE-CSE505), a 2-credit laboratory in Semester 5. The theory course is its same-semester corequisite, not an earlier prerequisite. The laboratory's own earlier prerequisites are Probability and Statistics (HCE-CSE303); Linear Algebra for Computing (HCE-CSE404).

## Learning objectives

- Distinguish supervised and unsupervised learning.
- Train and evaluate predictive models with suitable metrics.
- Identify overfitting and data leakage.
- Explain feature preparation and model validation.

## Unit structure

### Unit 1 — Learning Foundations

Topics: Learning paradigms; Features and labels; Training and test partitions.

Learning foundations distinguish supervised tasks with labels from unsupervised exploration of structure. Features describe observations, while labels specify the target when one is available. Training and test partitions separate model construction from later evaluation. The unit makes clear that success on examples used during learning is not sufficient evidence of generalization. Students describe the prediction problem before deciding whether a model family is appropriate.

### Unit 2 — Supervised Prediction

Topics: Linear regression; Logistic regression; Decision trees; Nearest neighbours.

Supervised prediction connects linear regression with numerical targets and logistic regression with class decisions. Decision trees and nearest neighbours offer contrasting ways to organize predictions. The emphasis is on what information a model uses and how its assumptions affect outputs. Classification and regression share the use of examples but produce different kinds of decisions. Feature preparation is interpreted in relation to the task, not treated as an unrelated preprocessing ritual.

### Unit 3 — Evaluation and Generalization

Topics: Cross-validation; Classification metrics; Bias and variance; Regularization.

Evaluation and generalization use cross-validation and classification metrics to inspect performance beyond a single score. Bias and variance provide a vocabulary for discussing underfitting and overfitting, while regularization controls excessive adaptation to particular training observations. Data leakage is examined as a failure of experimental design rather than a mysterious property of the model. Students compare the evidence supporting a result with the conclusion being claimed.

### Unit 4 — Unsupervised Learning

Topics: Clustering; Principal component analysis; Distance measures.

Unsupervised learning considers clustering and principal component analysis as different ways to explore observations without supplied prediction labels. Distance measures influence which samples appear related. Clusters summarize patterns in a chosen representation; they are not automatically meaningful categories. Dimensionality reduction changes the coordinate view and may preserve some relationships more clearly than others. Interpretation should therefore connect the numerical result with the question that motivated the analysis.

### Unit 5 — Neural Learning Introduction

Topics: Perceptron; Gradient descent; Basic multilayer networks; Responsible model use.

The neural learning introduction uses the perceptron and basic multilayer networks to connect predictions with adjustable parameters. Gradient descent supplies the idea of incremental improvement through an objective. The treatment prepares students for later Deep Learning without making this course a complete study of large neural architectures. Responsible model use includes explaining limitations and examining whether evaluation justifies applying a model to observations unlike those used for training.

## Expected learning outcomes

Students should be able to describe a learning task, select an appropriate evaluation approach, distinguish feature preparation from model fitting, and recognize unsupported claims caused by overfitting or leakage. NLP later applies predictive and representational ideas to language. Deep Learning develops multilayer neural methods more extensively. Those connections do not make the three courses interchangeable.

## Assessment reference

The course has 100 total marks: 40 internal assessment marks and 60 end-semester marks. Passing requires at least 50 total marks and at least 24 end-semester marks. Both conditions apply together; there is no separate internal minimum. Attendance eligibility remains a separate requirement. Internal assessment contains a 20-mark midterm and 20 marks of continuous coursework. Written end-semester assessment is used for this theory course. The associated laboratory uses the same marking scheme with practical demonstration and viva. Refer to Examination Regulations for grade bands and Attendance Policy for eligibility; no additional course-specific pass threshold is introduced here.
