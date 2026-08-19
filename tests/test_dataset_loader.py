from utils.dataset_loader import load_dataset


def test_dataset_loads():
    data = load_dataset()

    assert isinstance(data, list)
    assert len(data) > 0


def test_dataset_record_structure():
    data = load_dataset()

    required_fields = {
        "id",
        "claim",
        "evidence",
        "label",
        "source",
    }

    for record in data:
        assert required_fields.issubset(record.keys())


def test_dataset_labels():
    data = load_dataset()

    valid_labels = {"TRUE", "FALSE", "NOT_ENOUGH_EVIDENCE"}

    for record in data:
        assert record["label"] in valid_labels