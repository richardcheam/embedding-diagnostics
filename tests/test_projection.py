import numpy as np

from jepa_lens.diagnostics.projection import project_2d


def test_projection_returns_two_dimensions():
    rng = np.random.default_rng(0)
    points = project_2d(rng.normal(size=(200, 16)))
    assert len(points) == 200
    assert all(len(point) == 2 for point in points)


def test_projection_subsamples_to_max():
    rng = np.random.default_rng(0)
    points = project_2d(rng.normal(size=(5000, 16)), max_samples=100)
    assert len(points) == 100


def test_projection_is_json_serializable():
    import json

    rng = np.random.default_rng(0)
    points = project_2d(rng.normal(size=(50, 8)))
    json.dumps(points)


def test_collapsed_input_projects_to_tight_cluster():
    """A collapsed representation should occupy almost no area in 2D."""
    embeddings = np.ones((200, 16)) + np.random.default_rng(0).normal(scale=1e-9, size=(200, 16))
    points = np.array(project_2d(embeddings))
    assert points.std() < 1e-6


def test_handles_fewer_samples_than_dimensions():
    rng = np.random.default_rng(0)
    points = project_2d(rng.normal(size=(3, 64)))
    assert len(points) == 3
    assert all(len(point) == 2 for point in points)
