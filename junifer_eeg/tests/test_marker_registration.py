"""Test marker registration conflicts between old and new implementations."""


def test_marker_registration():
    """Test that marker registration doesn't have conflicts."""
    print("Testing marker registration...")

    try:
        # Import both implementations
        from junifer_eeg.markers.kolmogorov_complexity import (
            KolmogorovComplexity as OriginalKC,
        )
        from junifer_eeg.markers.kolmogorov_complexity_new.kolmogorov_complexity import (
            KolmogorovComplexity as NewKC,
        )

        print(f"Original KC class: {OriginalKC}")
        print(f"New KC class: {NewKC}")
        print(f"Original KC name: {OriginalKC.__name__}")
        print(f"New KC name: {NewKC.__name__}")

        # Check if they have different registration names
        print(
            f"Original KC registry name: {getattr(OriginalKC, '_marker_name', 'Not set')}"
        )
        print(
            f"New KC registry name: {getattr(NewKC, '_marker_name', 'Not set')}"
        )

        # Test instantiation
        orig_instance = OriginalKC()
        new_instance = NewKC()

        print(f"Original instance: {orig_instance}")
        print(f"New instance: {new_instance}")

        print("✅ Both implementations can be imported and instantiated")
        return True

    except Exception as e:
        print(f"❌ Registration conflict detected: {e}")
        return False


def test_output_type_method():
    """Test that get_output_type method works correctly."""
    print("\nTesting get_output_type method...")

    from junifer_eeg.markers.kolmogorov_complexity_new.kolmogorov_complexity import (
        KolmogorovComplexity,
    )

    # Test different aggregation combinations
    test_cases = [
        ({}, "timeseries"),
        ({"channel_aggregation_method": "mean"}, "vector"),
        ({"trial_aggregation_method": "mean"}, "vector"),
        (
            {
                "channel_aggregation_method": "mean",
                "trial_aggregation_method": "mean",
            },
            "scalar_table",
        ),
    ]

    for params, expected_type in test_cases:
        marker = KolmogorovComplexity(**params)
        actual_type = marker.get_output_type("EEG", "kolmogorovcomplexity")

        if actual_type == expected_type:
            print(f"  ✅ {params} -> {actual_type}")
        else:
            print(
                f"  ❌ {params} -> Expected {expected_type}, got {actual_type}"
            )
            return False

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("MARKER REGISTRATION AND OUTPUT TYPE TESTS")
    print("=" * 50)

    success = True
    success &= test_marker_registration()
    success &= test_output_type_method()

    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL REGISTRATION TESTS PASSED")
    else:
        print("❌ REGISTRATION ISSUES DETECTED")
    print("=" * 50)
