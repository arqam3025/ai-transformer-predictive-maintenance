from scripts.prepare_training_dataset import (
    GAS_COLUMNS,
    prepare_dataset,
)


EXPECTED_CLASSES = {"PD", "D1", "D2", "T1", "T2", "T3"}


def test_clean_dataset_has_expected_size():
    df = prepare_dataset()
    assert len(df) == 577


def test_all_gas_signatures_are_unique():
    df = prepare_dataset()

    assert len(df[GAS_COLUMNS].drop_duplicates()) == len(df)


def test_no_missing_values():
    df = prepare_dataset()

    assert not df.isna().any().any()


def test_no_negative_gas_measurements():
    df = prepare_dataset()

    assert (df[GAS_COLUMNS] >= 0).all().all()


def test_expected_fault_classes():
    df = prepare_dataset()

    assert set(df["fault_class"].unique()) == EXPECTED_CLASSES


def test_no_conflicting_labels_remain():
    df = prepare_dataset()

    label_counts = (
        df.groupby(GAS_COLUMNS)["fault_class"]
        .nunique()
    )

    assert label_counts.max() == 1


def test_sample_ids_are_unique():
    df = prepare_dataset()

    assert df["sample_id"].is_unique


def test_source_labels_are_preserved():
    df = prepare_dataset()

    assert "source_fault_label" in df.columns
    assert not df["source_fault_label"].isna().any()