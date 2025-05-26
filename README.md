# Feature Selection for Dummies

Selecting input features for a machine learning model may initially appear simple, but it quickly becomes a critical and complex step in the modeling process. Although few features might not capture the complexity of the problem, too many can complicate the learning process and negatively impact the model's interpretability. For this reason, choosing features precisely is not just a technical decision, but a strategic one.

As a data scientist, when selecting a set of features, I ask myself some questions:  
- Is this feature relevant to the problem I'm trying to solve?  
- Is this feature stable over time?  
- In classification problems, does the distribution differ across the different classes?  

Let's now connect each of these questions to appropriate methods for answering them.

---

## Is this feature relevant to the problem I'm trying to solve?

Feature importance from Random Forests is the most common method used to understand what matters. However, internally, Random Forest's feature importance is calculated based on how frequently a feature splits nodes in the trees. As a result, features with many categories, high variability, and non-monotonic relationships may receive higher feature importance scores, even if they do not have a greater actual impact on the model's performance.

Nowadays, one of the most accurate state-of-the-art methods is to calculate SHAP values (SHapley Additive exPlanations). SHAP values fairly distribute each feature's contribution to the prediction. Therefore, if a feature has a greater impact on the model, its average SHAP value, in absolute terms, will be correspondingly higher.

---

## Is this feature stable over time?

Correlation with the target is not everything. Even if a variable plays a key role in distinguishing the problem, it can become problematic if it changes significantly over time. In such cases, the model may learn the feature's behavior during training, but fail to generalize if that behavior shifts in production, turning the feature into a potential liability. Data drift can result from changes in user behavior, system updates, or external factors, all of which can lead to performance degradation over time.

**Best Practice:**  
Use a statistical test to assess whether the feature's data comes from the same distribution in both the training and test sets. Specifically, apply the **Chi-squared test** for categorical variables and the **Kolmogorov–Smirnov (KS) test** for continuous ones.  
Do the same feature in training and production belong to the same statistical distribution?

---
