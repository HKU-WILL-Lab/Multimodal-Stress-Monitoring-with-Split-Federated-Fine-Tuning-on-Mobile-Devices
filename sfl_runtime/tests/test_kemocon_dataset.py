from sfl_clean.kemocon_dataset import (
    SPLIT_SUBJECTS,
    participant_from_record_id,
    split_for_participant,
)


def test_participant_split_is_disjoint_and_complete() -> None:
    train = SPLIT_SUBJECTS["train"]
    validation = SPLIT_SUBJECTS["validation"]
    test = SPLIT_SUBJECTS["test"]
    assert len(train) == 22
    assert len(validation) == 3
    assert len(test) == 3
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert len(train | validation | test) == 28


def test_record_identifier_maps_to_one_split() -> None:
    assert participant_from_record_id("P4-W19") == 4
    assert split_for_participant(4) == "train"
    assert split_for_participant(12) == "validation"
    assert split_for_participant(25) == "test"
