# Changelog

## [0.2.1] - 2026-03-24

### Added
- Refreshed README with v0.2.0 project overview, badges, dependency guidance, and notebook-platform sections
- Rebuilt the quickstart notebook as a fuller end-to-end tutorial for setup, EDA, prep, visualization, ML, metrics, and text utilities

### Fixed
- `dsbro.about()` now shows the GitHub homepage and website as separate fields

## [0.2.0] - 2026-03-24

### Added
- Full EDA module: overview, profile, correlate, distribution, outliers, compare, drift
- Full preprocessing module: encode, scale, fill_missing, reduce_memory, target_encode, auto_preprocess
- Full visualization module with dark theme: bar, scatter, heatmap, confusion_matrix, roc_curve, feature_importance
- Full ML module: compare models, train, tune, blend, power_mean, oof_predict, adversarial_validation
- Metrics module: classification_report, regression_report, metric, competition_score
- Text module: clean_text, tokenize, ngrams, word_frequency, tfidf_features
- Help system: dsbro.help(), dsbro.about()
- Pretty printed outputs with clean formatting
- GitHub Actions CI/CD
- Contributing guidelines

### Fixed
- Metrics API alignment
- Homepage URL

## [0.1.0] - 2026-03-23

### Added
- Initial release
- IO module: tree, load, save, peek, fileinfo, find, merge_csvs
- Utils module: setup, seed, timer, gpu_info, system_info
