import pytest
import sublinear_cpp


def test_cms_initialization():
    # Valid initialization
    sketch = sublinear_cpp.CountMinSketch(width=1000, depth=5)
    assert sketch is not None

    # Invalid initialization should trigger the C++ std::invalid_argument (ValueError in Python)
    with pytest.raises(ValueError):
        sublinear_cpp.CountMinSketch(width=0, depth=5)


def test_add_and_estimate():
    sketch = sublinear_cpp.CountMinSketch(width=10000, depth=5)

    # Add a word and check its exact count
    sketch.add("kernel")
    assert sketch.estimate("kernel") == 1

    # Add it multiple times
    for _ in range(9):
        sketch.add("kernel")

    assert sketch.estimate("kernel") == 10

    # Unseen words should be 0
    assert sketch.estimate("ghost_word") == 0


def test_top_k_tracking():
    # width=10000, depth=5, track top 3
    sketch = sublinear_cpp.TopKCountMinSketch(width=10000, depth=5, k_size=3)

    # Insert with explicit frequencies
    for _ in range(100):
        sketch.add("gpu")
    for _ in range(50):
        sketch.add("cpu")
    for _ in range(10):
        sketch.add("ram")
    for _ in range(5):
        sketch.add("disk")  # Should be pushed out of Top 3

    top_items = sketch.get_top_k_items()

    # Verify the size is capped at k=3
    assert len(top_items) == 3

    # Verify exact ordering and frequencies (highest first)
    assert top_items[0] == ("gpu", 100)
    assert top_items[1] == ("cpu", 50)
    assert top_items[2] == ("ram", 10)


def test_save_and_load_binary(tmp_path):
    # tmp_path is a built-in pytest fixture that creates a safe temp directory
    filepath = str(tmp_path / "test_sketch.bin")

    # 1. Create and populate the original sketch
    original_sketch = sublinear_cpp.CountMinSketch(width=5000, depth=4)
    for _ in range(42):
        original_sketch.add("data")

    # Save it to disk
    original_sketch.save(filepath)

    # 2. Create a brand new, empty sketch with arbitrary dimensions
    # The load function should overwrite these dimensions with the saved ones
    loaded_sketch = sublinear_cpp.CountMinSketch(width=10, depth=2)
    loaded_sketch.load(filepath)

    # 3. Verify the state was perfectly restored
    assert loaded_sketch.estimate("data") == 42
    assert loaded_sketch.estimate("empty") == 0


def test_save_io_error():
    sketch = sublinear_cpp.CountMinSketch(width=100, depth=3)

    # Attempting to save to an invalid path should trigger std::runtime_error
    with pytest.raises(RuntimeError):
        sketch.save("/this/path/does/not/exist/sketch.bin")
