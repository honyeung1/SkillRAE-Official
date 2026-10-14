I am a data scientist, and I have a breast cancer classification project in ml_project/. The baseline model in train_baseline.py uses an SVM with suboptimal hyperparameters — no feature scaling, linear kernel, C=0.001 — achieving only about 90% accuracy on the Wisconsin Breast Cancer dataset with 569 samples and 30 features.

Please help me systematically optimize this model to achieve at least 96% test accuracy. I want you to:

1. First, run train_baseline.py to confirm baseline performance.

2. Try at least two different hyperparameter optimization approaches, such as exhaustive search and a smarter search method, over SVM hyperparameters including kernel, C, and gamma. Use proper feature scaling and cross-validation.

3. Compare the approaches and identify which one works better.

4. Train the final best model and save it as best_model.pkl.

5. Generate comparison.json containing: baseline_accuracy, results from both optimization methods including best_params, best_cv_score, and search_time_seconds, as well as best_method, best_params, final_test_accuracy, final_test_f1, and improvement_over_baseline.