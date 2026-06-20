def test_basic_imports():
    """Smoke test to ensure key modules import and basic symbols exist."""
    import sys
    from pathlib import Path
    # Ensure repo root is on sys.path for imports
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import preprocessDataset
    import train_models
    import evaluate_model
    import app

    assert hasattr(preprocessDataset, 'create_data_pipeline')
    assert hasattr(train_models, 'build_baseline_cnn')
    assert hasattr(evaluate_model, 'evaluate')
    assert hasattr(app, 'app')
