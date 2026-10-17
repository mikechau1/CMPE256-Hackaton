from consultant.recommenders.components import detect_series, score_fault_modes


def test_detect_series():
    assert detect_series("Our Catalyst 8300 keeps rebooting") == ["Catalyst 8300"]
    assert detect_series("ISR4331 and an ASR1001-X") == ["ISR 4000", "ASR 1000"]
    assert detect_series("our router is down") == []


def test_fault_modes_use_word_boundaries():
    names = [m["name"] for m in score_fault_modes("enterprise router has intermittent connectivity")]
    assert not any("Voice" in n for n in names)  # 'pri' inside 'enterprise' must not match
    assert names[0].startswith("Interface")


def test_fault_mode_weights_normalised():
    modes = score_fault_modes("router overheats and the fan is loud, then reloads")
    assert abs(sum(m["weight"] for m in modes) - 1.0) < 0.01
    assert all(m["matched_symptoms"] for m in modes)
