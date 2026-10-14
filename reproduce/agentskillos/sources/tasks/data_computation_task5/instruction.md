I am a biostatistics graduate student, and I need to complete a homework assignment on statistical hypothesis testing. Please perform hypothesis tests on the following 3 datasets and save results to hypothesis_tests.json and hypothesis_tests.xlsx.

Dataset 1 (Two-sample t-test): Control group [23.1, 22.5, 24.0, 21.8, 23.5, 22.9, 24.2, 23.0, 22.7, 23.8, 21.5, 24.1, 23.3, 22.6, 23.9]; Treatment group [26.3, 25.8, 27.1, 26.5, 25.2, 26.8, 27.3, 25.9, 26.1, 27.0, 25.5, 26.7, 26.0, 25.6, 26.4]. Test whether the treatment group mean is significantly greater than the control group (α=0.05).

Dataset 2 (Chi-square test of independence): Contingency table — Preference (A/B/C) vs Age group (Young/Middle/Old): [[30,25,15],[20,30,20],[10,15,35]]. Test whether preference and age are independent.

Dataset 3 (One-way ANOVA): Drug A [22.5,24.1,23.0,25.2,23.8,24.5], Drug B [26.3,25.8,27.1,26.5,25.2,26.8], Drug C [31.2,30.5,32.0,31.8,30.9,31.6]. Test whether the three group means differ significantly.

JSON output format — use exactly this structure:
{
  "t_test": {"test_name": "...", "statistic": ..., "p_value": ..., "conclusion": "reject or fail_to_reject", "effect_size": ..., "control_mean": ..., "treatment_mean": ...},
  "chi_square": {"test_name": "...", "statistic": ..., "p_value": ..., "conclusion": "...", "effect_size": ..., "df": ...},
  "anova": {"test_name": "...", "statistic": ..., "p_value": ..., "conclusion": "...", "effect_size": ..., "drug_a_mean": ..., "drug_b_mean": ..., "drug_c_mean": ...}
} The Excel file should have 3 sheets (T-Test, Chi-Square, ANOVA), each containing raw data and computed results.