NAME: JASWANTH REDDY REG NO: 24MAI0087

## Deep Learning – DA 2 – – Question Paper

The student has to execute the below algorithms and present them in the DA with the points mentioned below.

Question-DA2: Measuring Bias and Fairness Designing Fair AI systems, Legal and ethical considerations Mitigating Bias and implementing Cultural implications of AI using Real time scenario.

## Points to cover in DA:

-  Theoretical presentation of each topic/sub-topic mentioned in the question
-  Strengths and weaknesses of the algorithms/ techniques
-  Algorithms, flow charts, and Coding
-  Execution results including Screenshots like input and outputs for each algorithm
-  Max. number of pages: As per the required, no limitation.
-  Programming Language: Any programming language for coding can be used as per the student's choice
-  Last date for submission in VTOP (Check in VTOP)
-  Give proper conclusion and references
-  No plagiarism

## 1.1 Measuring Bias and Fairness

Bias in artificial intelligence systems refers to systematic and unfair discrimination that leads to unjust or unequal outcomes, particularly against certain groups defined by characteristics such as gender, race, age, or income. Measuring bias is the first essential step in creating ethical, trustworthy, and fair AI systems.

Bias is typically detected by observing how predictions vary across different groups. If a machine learning model favors one group more than another in terms of accuracy, outcomes, or benefits, it is considered biased.

## Common metrics for measuring bias:

-  Selection Rate: Measures the rate at which different groups receive positive outcomes.
-  True Positive Rate (TPR): Measures the percentage of actual positives correctly identified, segmented by group.
-  False Positive Rate (FPR): Measures the percentage of actual negatives incorrectly identified as positive, segmented by group.
-  Disparate Impact: Measures the ratio of outcomes between protected and unprotected groups, ideally close to 1.

These metrics help quantify whether the model ' s decisions are fair and balanced across different demographics.

## 1.2 Designing Fair AI Systems

Designing fair AI systems involves a multi-stage strategy that starts from data acquisition and ends at model deployment. At every stage, potential bias must be identified and mitigated. Critical considerations in the design process include:

-  Balanced and Representative Datasets: Data should be collected in a way that represents all classes and groups to avoid historical discrimination.
-  Feature Selection: Care must be taken to avoid using sensitive or proxy features (like zip code as a proxy for race).
-  Algorithm Choice: Use fairness-aware algorithms or include fairness constraints.
-  Evaluation Metrics: Evaluate both accuracy and fairness metrics.
-  Post -deployment Monitoring: AI should be monitored for drift in fairness over time.

One popular strategy is to use pre-processing , in -processing, and post-processing fairness techniques. These approaches remove or reduce bias from datasets, models, and predictions respectively.

## 1.3 Legal and Ethical Considerations

Legal frameworks such as the European Union's General Data Protection Regulation (GDPR), the Equal Credit Opportunity Act, and industry-specific compliance laws (like HIPAA in healthcare) demand fair, transparent, and accountable AI systems.

From an ethical standpoint, AI must align with fundamental principles:

-  Autonomy: Respecting user freedom and informed consent.
-  Justice: Avoiding discrimination or bias.
-  Non -maleficence: Preventing harm caused by biased decisions.
-  Accountability: Ensuring developers and deployers are responsible for model actions.

NAME: JASWANTH REDDY

REG NO: 24MAI0087

Ethical AI is not only a moral responsibility but also a practical requirement for gaining public trust and avoiding legal consequences.

## 1.4 Mitigating Bias and Cultural Implications Using Real-Time Scenario

Bias mitigation can be illustrated through real-time scenarios. Consider a hiring algorithm trained on historical company data that primarily includes male employees. This model may learn to favor male applicants, continuing a cycle of gender discrimination. Similarly, in financial lending, models may learn to reject loan applications from individuals in historically underserved communities due to biased historical trends.

To mitigate such biases, developers can:

-  Balance the dataset by resampling or reweighting.
-  Introduce fairness constraints during training.
-  Post -process the predictions to ensure equality of outcomes.

Understanding cultural contexts is essential. A model that is fair in one country may be considered biased in another. Culturally sensitive design involves collaborating with domain experts, sociologists, and community stakeholders to ensure ethical appropriateness.

## 2. Strengths and Weaknesses

## Strengths

-  Allows identification of hidden discrimination.
-  Improves transparency, accountability, and public trust in AI.
-  Legally y and ethically aligns AI development with real-world regulations.
-  Tools like Fairlearn, SHAP, and LIME make fairness quantifiable and explainable.

## Weaknesses

-  Definitions of fairness are often subjective and vary across domains.
-  Balancing fairness may reduce model accuracy.
-  Requires interdisciplinary collaboration (technical, legal, ethical).
-  Proxy bias is difficult to detect and remove completely.

NAME: JASWANTH REDDY

REG NO: 24MAI0087

## 3. Flowchart of AI Bias Mitigation

```
mathematica CopyEdit Start ↓ Load Dataset → Clean and Preprocess Data → Identify Sensitive Features ↓ Train Initial Model → Evaluate Accuracy and Fairness ↓ Visualize and Explain Bias using LIME / SHAP ↓ Apply Fairness Mitigation (Exponentiated Gradient, etc.) ↓ Re -evaluate Fairness Metrics → Visualize Results ↓ Deploy Model or Repeat Process ↓ End
```

## 4. Python Implementation

The full Python code was already provided earlier in our conversation. Here's what it includes:

-  Logistic Regression Model trained on the Adult Income dataset.
-  Measurement of bias using Fairlearn with selection rate and true/false positive rates.
-  Fairness mitigation using ExponentiatedGradient with a fairness constraint (DemographicParity).
-  Local interpretability with LIME to understand feature-level decisions.
-  Global and local interpretability using SHAP to understand feature contributions and potential proxies for bias.

NAME: JASWANTH REDDY REG NO: 24MAI0087

NAME: JASWANTH REDDY

REG NO: 24MAI0087

<!-- image -->

## SOURCE CODE:

<!-- image -->

<!-- image -->

NAME: JASWANTH REDDY

REG NO: 24MAI0087

<!-- image -->

<!-- image -->

<!-- image -->

## NAME: JASWANTH REDDY

REG NO: 24MAI0087

<!-- image -->

<!-- image -->

## NAME: JASWANTH REDDY REG NO: 24MAI0087

<!-- image -->

## Explanation of Implementation for Measuring and Mitigating AI Bias

This implementation is based on detecting and mitigating bias in a real-world classification task using the Adult Income dataset. The goal is to evaluate how a machine learning model can unfairly treat different demographic groups and how such unfairness can be mitigated using fairness-aware algorithms.

## Dataset Overview

The Adult Income dataset from UCI contains census records and aims to predict whether a person's income exceeds $50,000 per year. It includes features such as age, education level, hours worked per week, occupation, and sex. In this context, gender is treated as a sensitive attribute to evaluate fairness.

## Data Preprocessing

The dataset is cleaned to remove missing values. The income column is converted into binary labels: 1 for income greater than $50K and 0 for income less than or equal to $50K. Only a subset of numerical features such as age, education level, and hours worked per week is used for training the model. Gender is not included as a feature in training but is used as a sensitive attribute for fairness evaluation.

## Splitting the Data

The dataset is divided into training and testing sets. Along with the feature and target variables, the gender column is separated out so that it can later be used to evaluate whether the model behaves differently for different gender groups.

## Model Training

A logistic regression model is trained using the selected features. This model is used to predict whether an individual earns more than $50K per year. Even though the gender feature is not directly used in the model, it is possible for the model to learn biases indirectly from proxy variables that correlate with gender.

## Fairness Evaluation (Before Mitigation)

To assess whether the model is biased, fairness metrics are calculated separately for male and female groups. The metrics used are:

-  Selection Rate: The percentage of individuals in each group predicted to earn more than $50K.
-  True Positive Rate: The proportion of individuals who actually earn more than $50K and are correctly identified.
-  False Positive Rate: The proportion of individuals who earn less than $50K but are incorrectly predicted to earn more.

If these values are significantly different across gender groups, the model is considered biased.

## Fairness Mitigation Technique

A bias mitigation technique called Exponentiated Gradient is applied. This method adjusts the m odel during training by introducing a fairness constraint known as Demographic Parity. The goal of this constraint is to ensure that the model gives equal selection rates (i.e., predicted positive outcomes) across all gender groups, regardless of their true labels.

This technique involves training a model that balances fairness and accuracy, often leading to improved equity in outcomes.

## Fairness Evaluation (After Mitigation)

After applying the mitigation technique, the model ' s fairness is reassessed using the same metrics. The new results show whether the disparity between gender groups has reduced. Ideally, the selection rate, true positive rate, and false positive rate should become more aligned across groups.

## Visualization Using Scatter Plot

To visualize the effectiveness of the mitigation, a scatter plot is created. It shows the selection rate for each gender both before and after the mitigation process. This visual comparison

NAME: JASWANTH REDDY REG NO: 24MAI0087

NAME: JASWANTH REDDY REG NO: 24MAI0087

highlights how the model's fairness has improved. The closer the points are to each other, the more balanced the model's outcomes are across gender groups.

## Conclusion

The process demonstrates that even a model trained without explicitly biased features can still exhibit unfair behavior due to correlations in the data. By applying fairness evaluation metrics and mitigation strategies, this unfairness can be detected and significantly reduced.

This approach is essential in real-world applications where decisions made by machine learning models directly impact people ' s lives, such as in hiring, lending, or healthcare. Mitigating bias not only ensures compliance with ethical and legal standards but also enhances the trustworthiness and inclusivity of AI systems.