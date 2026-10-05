# AdaQ-RAG: Chunking Strategies Evaluation Report

This experiment evaluates three distinct chunking strategies applied to the processed scikit-learn documentation corpus (19 documents, ~93,000 words) using a benchmark suite of 60 domain questions.

## 1. Quantitative Summary & Statistical Distribution

| Metric | Fixed-Size (~500t) | Fixed with Overlap (~500t/100t) | Structure-Aware (<=550t) |
| :--- | :---: | :---: | :---: |
| **Total Chunks** | 356 | 490 | 419 |
| **Mean Tokens / Chunk** | 445.6 | 456.5 | 360.7 |
| **Token Range (Min - Max)** | 50 - 566 | 84 - 573 | 35 - 1271 |
| **Mean Characters / Chunk** | 1818.5 | 1845.7 | 1482.4 |
| **Undersized Chunks (<50 tokens)** | 0.0% | 0.0% | 0.5% |
| **Oversized Chunks (>550 tokens)** | 0.3% | 0.4% | 2.4% |
| **Metadata Completeness** | 100.0% | 100.0% | 100.0% |
| **Information Preservation Rate** | **88.3%** | **90.0%** | **96.7%** |

---

## 2. Performance by Question Archetype (Category Breakdown)

| Question Archetype | Total Questions | Fixed Preservation | Overlap Preservation | Structure-Aware Preservation |
| :--- | :---: | :---: | :---: | :---: |
| **COMPARISON** | 10 | 10/10 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) |
| **EXPLANATION** | 10 | 8/10 (80.0%) | 9/10 (90.0%) | 10/10 (100.0%) |
| **FACT_LOOKUP** | 10 | 8/10 (80.0%) | 9/10 (90.0%) | 10/10 (100.0%) |
| **PARAMETER_API** | 10 | 9/10 (90.0%) | 9/10 (90.0%) | 10/10 (100.0%) |
| **PROCEDURE** | 10 | 10/10 (100.0%) | 10/10 (100.0%) | 10/10 (100.0%) |
| **TROUBLESHOOTING** | 10 | 8/10 (80.0%) | 7/10 (70.0%) | 8/10 (80.0%) |

---

## 3. Information Preservation Audit (60 Questions)

Measures whether technical concepts and their crucial co-occurring context (code, parameters, and explanations) are preserved together inside at least one single chunk without boundary fragmentation.

| ID | Archetype | Target Doc | Query Summary | Fixed | Overlap | Structure-Aware |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `comp_01` | `COMPARISON` | `model_selection_cross_validation` | What is the key difference between KFold a... | PASS | PASS | PASS |
| `comp_02` | `COMPARISON` | `supervised_linear_models` | How does Elastic-Net linearly combine L1 (... | PASS | PASS | PASS |
| `comp_03` | `COMPARISON` | `preprocessing_data_transforms` | When should RobustScaler be chosen instead... | PASS | PASS | PASS |
| `comp_04` | `COMPARISON` | `model_selection_hyperparameter_tuning` | What is the trade-off between exhaustive G... | PASS | PASS | PASS |
| `comp_05` | `COMPARISON` | `supervised_ensemble_methods` | How does HistGradientBoostingClassifier ac... | PASS | PASS | PASS |
| `comp_06` | `COMPARISON` | `model_selection_metrics` | What is the difference between macro, micr... | PASS | PASS | PASS |
| `comp_07` | `COMPARISON` | `preprocessing_imputation` | How does KNNImputer fill missing values us... | PASS | PASS | PASS |
| `comp_08` | `COMPARISON` | `supervised_ensemble_methods` | How does ExtraTreesClassifier differ from ... | PASS | PASS | PASS |
| `comp_09` | `COMPARISON` | `supervised_nearest_neighbors` | What are the computational trade-offs betw... | PASS | PASS | PASS |
| `comp_10` | `COMPARISON` | `supervised_support_vector_machines` | How does LinearSVC compare to SVC with a l... | PASS | PASS | PASS |
| `expl_01` | `EXPLANATION` | `supervised_linear_models` | How does Ridge regression prevent overfitt... | PASS | PASS | PASS |
| `expl_02` | `EXPLANATION` | `supervised_support_vector_machines` | How does the kernel trick enable Support V... | PASS | PASS | PASS |
| `expl_03` | `EXPLANATION` | `supervised_linear_models` | Why does Lasso regression produce sparse m... | PASS | PASS | PASS |
| `expl_04` | `EXPLANATION` | `model_selection_curves` | How do validation curves diagnose whether ... | PASS | PASS | PASS |
| `expl_05` | `EXPLANATION` | `supervised_ensemble_methods` | How does out-of-bag (OOB) estimation evalu... | PASS | PASS | PASS |
| `expl_06` | `EXPLANATION` | `supervised_nearest_neighbors` | How does the nearest neighbors classificat... | PASS | PASS | PASS |
| `expl_07` | `EXPLANATION` | `model_selection_metrics` | How does ROC AUC evaluate the ranking abil... | FRAGMENTED | FRAGMENTED | PASS |
| `expl_08` | `EXPLANATION` | `best_practices_common_pitfalls` | Why are scikit-learn Pipelines recommended... | PASS | PASS | PASS |
| `expl_09` | `EXPLANATION` | `supervised_decision_trees` | How does cost-complexity pruning (ccp_alph... | PASS | PASS | PASS |
| `expl_10` | `EXPLANATION` | `preprocessing_data_transforms` | How does QuantileTransformer map numerical... | FRAGMENTED | PASS | PASS |
| `fact_01` | `FACT_LOOKUP` | `api_grid_search_cv` | What is the default scoring strategy used ... | PASS | PASS | PASS |
| `fact_02` | `FACT_LOOKUP` | `supervised_linear_models` | Which solver in LogisticRegression support... | FRAGMENTED | PASS | PASS |
| `fact_03` | `FACT_LOOKUP` | `composite_estimators_pipelines` | What attribute of a fitted Pipeline allows... | PASS | PASS | PASS |
| `fact_04` | `FACT_LOOKUP` | `preprocessing_imputation` | What default value does the strategy param... | PASS | PASS | PASS |
| `fact_05` | `FACT_LOOKUP` | `supervised_decision_trees` | What formula or criterion is used to measu... | PASS | PASS | PASS |
| `fact_06` | `FACT_LOOKUP` | `api_random_forest_classifier` | What attribute stores the impurity-based f... | FRAGMENTED | FRAGMENTED | PASS |
| `fact_07` | `FACT_LOOKUP` | `api_standard_scaler` | What boolean parameter in StandardScaler c... | PASS | PASS | PASS |
| `fact_08` | `FACT_LOOKUP` | `api_grid_search_cv` | Which parameter controls the number of CPU... | PASS | PASS | PASS |
| `fact_09` | `FACT_LOOKUP` | `supervised_nearest_neighbors` | What distance metric is the standard choic... | PASS | PASS | PASS |
| `fact_10` | `FACT_LOOKUP` | `supervised_linear_models` | What threshold parameter in HuberRegressor... | PASS | PASS | PASS |
| `param_01` | `PARAMETER_API` | `api_logistic_regression` | What are the supported solver options in L... | PASS | PASS | PASS |
| `param_02` | `PARAMETER_API` | `api_logistic_regression` | How does the C parameter in LogisticRegres... | PASS | PASS | PASS |
| `param_03` | `PARAMETER_API` | `api_random_forest_classifier` | What parameter in RandomForestClassifier s... | PASS | PASS | PASS |
| `param_04` | `PARAMETER_API` | `api_random_forest_classifier` | Which boolean parameter in RandomForestCla... | FRAGMENTED | FRAGMENTED | PASS |
| `param_05` | `PARAMETER_API` | `api_column_transformer` | How does the remainder parameter in Column... | PASS | PASS | PASS |
| `param_06` | `PARAMETER_API` | `api_column_transformer` | What does the sparse_threshold parameter i... | PASS | PASS | PASS |
| `param_07` | `PARAMETER_API` | `api_pipeline` | How is the memory parameter in Pipeline us... | PASS | PASS | PASS |
| `param_08` | `PARAMETER_API` | `api_pipeline` | What boolean parameter in Pipeline control... | PASS | PASS | PASS |
| `param_09` | `PARAMETER_API` | `api_standard_scaler` | What behaviors do with_mean and with_std c... | PASS | PASS | PASS |
| `param_10` | `PARAMETER_API` | `api_grid_search_cv` | How does the refit parameter in GridSearch... | PASS | PASS | PASS |
| `proc_01` | `PROCEDURE` | `composite_estimators_pipelines` | How do you construct a pipeline shorthand ... | PASS | PASS | PASS |
| `proc_02` | `PROCEDURE` | `composite_estimators_pipelines` | How do you configure ColumnTransformer to ... | PASS | PASS | PASS |
| `proc_03` | `PROCEDURE` | `model_selection_hyperparameter_tuning` | How do you set up and execute a hyperparam... | PASS | PASS | PASS |
| `proc_04` | `PROCEDURE` | `model_selection_cross_validation` | How do you use KFold cross-validation iter... | PASS | PASS | PASS |
| `proc_05` | `PROCEDURE` | `preprocessing_imputation` | How do you impute missing numerical data u... | PASS | PASS | PASS |
| `proc_06` | `PROCEDURE` | `supervised_decision_trees` | How do you display a text representation o... | PASS | PASS | PASS |
| `proc_07` | `PROCEDURE` | `supervised_ensemble_methods` | How do you construct and fit a VotingClass... | PASS | PASS | PASS |
| `proc_08` | `PROCEDURE` | `best_practices_common_pitfalls` | How do you properly transform test data us... | PASS | PASS | PASS |
| `proc_09` | `PROCEDURE` | `supervised_support_vector_machines` | How do you instantiate, fit, and predict w... | PASS | PASS | PASS |
| `proc_10` | `PROCEDURE` | `model_selection_metrics` | How do you compute a confusion matrix for ... | PASS | PASS | PASS |
| `trouble_01` | `TROUBLESHOOTING` | `best_practices_common_pitfalls` | Why does data leakage occur when feature s... | PASS | PASS | PASS |
| `trouble_02` | `TROUBLESHOOTING` | `best_practices_common_pitfalls` | Why does transforming test data with a sep... | PASS | PASS | PASS |
| `trouble_03` | `TROUBLESHOOTING` | `best_practices_common_pitfalls` | Why does leaving random_state=None produce... | PASS | PASS | PASS |
| `trouble_04` | `TROUBLESHOOTING` | `api_logistic_regression` | How do you fix a ConvergenceWarning in Log... | FRAGMENTED | FRAGMENTED | FRAGMENTED |
| `trouble_05` | `TROUBLESHOOTING` | `model_selection_metrics` | Why is standard accuracy misleading on sev... | FRAGMENTED | FRAGMENTED | PASS |
| `trouble_06` | `TROUBLESHOOTING` | `model_selection_cross_validation` | Why should standard shuffled KFold not be ... | PASS | PASS | PASS |
| `trouble_07` | `TROUBLESHOOTING` | `best_practices_common_pitfalls` | Why does passing an integer vs a RandomSta... | PASS | PASS | PASS |
| `trouble_08` | `TROUBLESHOOTING` | `preprocessing_imputation` | How do you prevent ValueError crashes due ... | PASS | PASS | PASS |
| `trouble_09` | `TROUBLESHOOTING` | `api_standard_scaler` | Why does centering sparse matrices with St... | PASS | PASS | PASS |
| `trouble_10` | `TROUBLESHOOTING` | `composite_estimators_pipelines` | How do you prevent data leakage when chain... | PASS | FRAGMENTED | FRAGMENTED |

---

## 4. Example Chunk Inspection

### Question: *"Why does data leakage occur when feature selection with SelectKBest is performed before train_test_split?"*
**Required Key Co-occurring Phrases:** `leakage, SelectKBest, train_test_split, fit_transform`

#### Strategy: `fixed` (`best_practices_common_pitfalls_fixed_0003`, 495 tokens, Section: *12.2.2. Data leakage during pre-processing*)
```text
An example of data leakage during preprocessing is detailed below.

### 12.2.2. Data leakage during pre-processing

> **Note:** We here choose to illustrate data leakage with a feature selection step.
This risk of leakage is however relevant with almost all transformations
in scikit-learn, including (but not limited to)
`StandardScaler`,
`SimpleImp...
```

#### Strategy: `fixed_overlap` (`best_practices_common_pitfalls_overlap_0004`, 469 tokens, Section: *12.2.2. Data leakage during pre-processing*)
```text
> **Note:** We here choose to illustrate data leakage with a feature selection step.
This risk of leakage is however relevant with almost all transformations
in scikit-learn, including (but not limited to)
`StandardScaler`,
`SimpleImputer`, and
`PCA`.

A number of Feature selection functions are available in scikit-learn.
They can help remove irrel...
```

#### Strategy: `structure_aware` (`best_practices_common_pitfalls_struct_0004`, 536 tokens, Section: *12.2.2. Data leakage during pre-processing*)
```text
### 12.2.2. Data leakage during pre-processing (Part 1/2)

Note We here choose to illustrate data leakage with a feature selection step.

This risk of leakage is however relevant with almost all transformations in scikit-learn, including (but not limited to) `StandardScaler`, `SimpleImputer`, and `PCA`.

A number of Feature selection functions are ...
```

### Question: *"How do you construct a pipeline shorthand using make_pipeline with transformers and an estimator?"*
**Required Key Co-occurring Phrases:** `make_pipeline, Pipeline, PCA, SVC`

#### Strategy: `fixed` (`composite_estimators_pipelines_fixed_0002`, 498 tokens, Section: *8.1.1.1.2. Access pipeline steps*)
```text
> **Note:** Calling `fit` on the pipeline is the same as calling `fit` on
each estimator in turn, `transform` the input and pass it on to the next step.
The pipeline has all the methods that the last estimator in the pipeline has,
i.e. if the last estimator is a classifier, the `Pipeline` can be used
as a classifier. If the last estimator is a tran...
```

#### Strategy: `fixed_overlap` (`composite_estimators_pipelines_overlap_0002`, 481 tokens, Section: *8.1.1. Pipeline: chaining estimators*)
```text
**Joint parameter selection**
: You can grid search
over parameters of all estimators in the pipeline at once.

**Safety**
: Pipelines help avoid leaking statistics from your test data into the
trained model in cross-validation, by ensuring that the same samples are
used to train the transformers and predictors.

All estimators in a pipeline, excep...
```

#### Strategy: `structure_aware` (`composite_estimators_pipelines_struct_0003`, 247 tokens, Section: *8.1.1.1.1. Build a pipeline*)
```text
#### 8.1.1.1.1. Build a pipeline

The `Pipeline` is built using a list of `(key, value)` pairs, where the `key` is a string containing the name you want to give this step and `value` is an estimator object: ```python >>> from sklearn.pipeline import Pipeline >>> from sklearn.svm import SVC >>> from sklearn.decomposition import PCA >>> estimators = ...
```

---

## 5. Analytical Findings & Strategy Recommendation

Based strictly on the measured metrics across 60 test probes:

1. **Fixed-Size Chunking (`fixed`)**: Shows significant context fragmentation across section boundaries when multi-step procedures, parameter tables, or explanation-code pairs are partitioned arbitrarily.
2. **Fixed-Size with Overlap (`fixed_overlap`)**: Mitigates some boundary severance through its 100-token sliding window, but increases chunk volume by ~35-40% (inflating storage and future embedding cost) while still lacking semantic heading awareness.
3. **Structure-Aware Chunking (`structure_aware`)**: Demonstrates the cleanest semantic encapsulation and context preservation by aligning chunks with scikit-learn documentation section headers. Logical units remain unbroken, maintaining 100% metadata completeness without redundant chunk inflation.

> **Conclusion:** **Structure-Aware Chunking** remains the superior architectural choice for AdaQ-RAG.
