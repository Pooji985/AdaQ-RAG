"""Curated documentation sources registry for scikit-learn corpus."""

from adaq_rag.ingestion.models import SourceDefinition

CURATED_SOURCES: list[SourceDefinition] = [
    # 1. Supervised Learning
    SourceDefinition(
        doc_id="supervised_linear_models",
        url="https://scikit-learn.org/stable/modules/linear_model.html",
        category="supervised_learning",
        description="Linear regression, Ridge, Lasso, ElasticNet, Logistic Regression, and SGD algorithms.",
    ),
    SourceDefinition(
        doc_id="supervised_support_vector_machines",
        url="https://scikit-learn.org/stable/modules/svm.html",
        category="supervised_learning",
        description="Support Vector Classification (SVC), SVR, mathematical formulations, and kernel parameters.",
    ),
    SourceDefinition(
        doc_id="supervised_decision_trees",
        url="https://scikit-learn.org/stable/modules/tree.html",
        category="supervised_learning",
        description="Decision Trees for classification and regression, splitting criteria, and tree pruning.",
    ),
    SourceDefinition(
        doc_id="supervised_ensemble_methods",
        url="https://scikit-learn.org/stable/modules/ensemble.html",
        category="supervised_learning",
        description="Random Forests, Gradient Boosting, HistGradientBoosting, Voting, and Stacking ensembles.",
    ),
    SourceDefinition(
        doc_id="supervised_nearest_neighbors",
        url="https://scikit-learn.org/stable/modules/neighbors.html",
        category="supervised_learning",
        description="k-Nearest Neighbors algorithms, metric spaces, classification, and regression.",
    ),
    # 2. Model Selection and Evaluation
    SourceDefinition(
        doc_id="model_selection_cross_validation",
        url="https://scikit-learn.org/stable/modules/cross_validation.html",
        category="model_selection_evaluation",
        description="Cross-validation iterators (KFold, StratifiedKFold, TimeSeriesSplit) and scoring.",
    ),
    SourceDefinition(
        doc_id="model_selection_hyperparameter_tuning",
        url="https://scikit-learn.org/stable/modules/grid_search.html",
        category="model_selection_evaluation",
        description="GridSearchCV, RandomizedSearchCV, and HalvingGridSearchCV hyperparameter optimization.",
    ),
    SourceDefinition(
        doc_id="model_selection_metrics",
        url="https://scikit-learn.org/stable/modules/model_evaluation.html",
        category="model_selection_evaluation",
        description="Classification, regression, clustering, and ranking metrics and scoring parameters.",
    ),
    SourceDefinition(
        doc_id="model_selection_curves",
        url="https://scikit-learn.org/stable/modules/learning_curve.html",
        category="model_selection_evaluation",
        description="Validation curves and learning curves for diagnostic assessment of model bias and variance.",
    ),
    # 3. Pipelines and Composite Estimators
    SourceDefinition(
        doc_id="composite_estimators_pipelines",
        url="https://scikit-learn.org/stable/modules/compose.html",
        category="pipelines_and_composites",
        description="Pipelines, ColumnTransformer, FeatureUnion, and composite estimator chaining.",
    ),
    # 4. Preprocessing and Dataset Transformations
    SourceDefinition(
        doc_id="preprocessing_data_transforms",
        url="https://scikit-learn.org/stable/modules/preprocessing.html",
        category="preprocessing_and_transforms",
        description="StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler, and non-linear transforms.",
    ),
    SourceDefinition(
        doc_id="preprocessing_imputation",
        url="https://scikit-learn.org/stable/modules/impute.html",
        category="preprocessing_and_transforms",
        description="Missing value imputation techniques: SimpleImputer, KNNImputer, and IterativeImputer.",
    ),
    # 5. Common Pitfalls and Recommended Practices
    SourceDefinition(
        doc_id="best_practices_common_pitfalls",
        url="https://scikit-learn.org/stable/common_pitfalls.html",
        category="common_pitfalls",
        description="Data leakage, evaluation pitfalls, inconsistent preprocessing, and randomness management.",
    ),
    # 6. Selected High-Value API References
    SourceDefinition(
        doc_id="api_logistic_regression",
        url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        category="api_reference",
        description="API reference specification for sklearn.linear_model.LogisticRegression.",
    ),
    SourceDefinition(
        doc_id="api_random_forest_classifier",
        url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
        category="api_reference",
        description="API reference specification for sklearn.ensemble.RandomForestClassifier.",
    ),
    SourceDefinition(
        doc_id="api_pipeline",
        url="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html",
        category="api_reference",
        description="API reference specification for sklearn.pipeline.Pipeline.",
    ),
    SourceDefinition(
        doc_id="api_column_transformer",
        url="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html",
        category="api_reference",
        description="API reference specification for sklearn.compose.ColumnTransformer.",
    ),
    SourceDefinition(
        doc_id="api_standard_scaler",
        url="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
        category="api_reference",
        description="API reference specification for sklearn.preprocessing.StandardScaler.",
    ),
    SourceDefinition(
        doc_id="api_grid_search_cv",
        url="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html",
        category="api_reference",
        description="API reference specification for sklearn.model_selection.GridSearchCV.",
    ),
]


def get_curated_sources() -> list[SourceDefinition]:
    """Return a copy of the curated documentation source definitions."""
    return list(CURATED_SOURCES)
