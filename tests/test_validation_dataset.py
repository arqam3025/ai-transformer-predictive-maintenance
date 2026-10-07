from scripts.build_validation_dataset import build_dataset


GAS_COLUMNS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]


def test_dataset_contains_25_transformers():
    df = build_dataset()
    assert len(df) == 25


def test_transformer_ids_are_unique():
    df = build_dataset()
    assert df["transformer_id"].is_unique


def test_gas_measurements_are_complete_and_non_negative():
    df = build_dataset()

    assert not df[GAS_COLUMNS].isna().any().any()
    assert (df[GAS_COLUMNS] >= 0).all().all()


def test_original_fault_descriptions_are_preserved():
    df = build_dataset()

    assert not df["fault_description"].isna().any()
    assert (df["fault_description"].str.len() > 0).all()


def test_combined_faults_are_not_silently_relabelled():
    df = build_dataset()

    assert "T3+D2" in set(df["fault_class"])
    assert "T2+D1" in set(df["fault_class"])


def test_known_source_measurement_is_preserved():
    df = build_dataset()

    transformer_21 = df.loc[df["transformer_id"] == 21].iloc[0]

    assert transformer_21["H2"] == 120
    assert transformer_21["CH4"] == 79
    assert transformer_21["C2H6"] == 50
    assert transformer_21["C2H4"] == 390
    assert transformer_21["C2H2"] == 8100
    assert transformer_21["fault_class"] == "D2"