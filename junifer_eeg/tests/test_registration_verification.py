"""Test script to verify marker registration names match between original and refactored versions."""

from junifer.markers import _MARKER_REGISTRY

from junifer_eeg.markers.time_locked_contrast import (
    TimeLockedContrast as OriginalTimeLockedContrast,
)
from junifer_eeg.markers.time_locked_new.time_locked_contrast import (
    TimeLockedContrast as RefactoredTimeLockedContrast,
)

# Import refactored implementations
from junifer_eeg.markers.time_locked_new.time_locked_topography import (
    TimeLockedTopography as RefactoredTimeLockedTopography,
)

# Import original implementations
from junifer_eeg.markers.time_locked_topography import (
    TimeLockedTopography as OriginalTimeLockedTopography,
)


def test_marker_registration():
    """Test that refactored markers register with same names as originals."""
    print("Testing marker registration names...")

    # Check original marker registration
    print("Original markers in registry:")
    for marker_class in [
        OriginalTimeLockedTopography,
        OriginalTimeLockedContrast,
    ]:
        marker_name = marker_class.__name__
        if marker_name in _MARKER_REGISTRY:
            print(f"  ✅ {marker_name}: Registered")
        else:
            print(f"  ❌ {marker_name}: NOT REGISTERED")

    # Check refactored marker registration
    print("\nRefactored markers in registry:")
    for marker_class in [
        RefactoredTimeLockedTopography,
        RefactoredTimeLockedContrast,
    ]:
        marker_name = marker_class.__name__
        if marker_name in _MARKER_REGISTRY:
            print(f"  ✅ {marker_name}: Registered")
        else:
            print(f"  ❌ {marker_name}: NOT REGISTERED")

    # Check for conflicts (both registered with same name)
    print("\nChecking for registration conflicts:")
    conflict_markers = []
    for marker_class in [
        OriginalTimeLockedTopography,
        OriginalTimeLockedContrast,
        RefactoredTimeLockedTopography,
        RefactoredTimeLockedContrast,
    ]:
        marker_name = marker_class.__name__
        if marker_name in _MARKER_REGISTRY:
            registered_class = _MARKER_REGISTRY[marker_name]
            if registered_class != marker_class:
                print(
                    f"  ⚠️  {marker_name}: Registered class differs from expected"
                )
                print(f"      Expected: {marker_class}")
                print(f"      Registered: {registered_class}")
                conflict_markers.append(marker_name)
            else:
                print(
                    f"  ✅ {marker_name}: Registration matches expected class"
                )

    return len(conflict_markers) == 0


def test_marker_instantiation():
    """Test that both original and refactored markers can be instantiated."""
    print("\nTesting marker instantiation...")

    # Test parameters
    topo_params = {
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": ["scalp"],
        "channel_aggregation_method": "mean",
        "trial_aggregation_method": "mean",
        "equipment": "egi256",
    }

    contrast_params = {
        "condition_a": "condition_a",
        "condition_b": "condition_b",
        "tmin": 0.1,
        "tmax": 0.3,
        "rois": ["scalp"],
        "channel_aggregation_method": "mean",
        "trial_aggregation_method": "mean",
        "equipment": "egi256",
    }

    # Test original markers
    try:
        _ = OriginalTimeLockedTopography(**topo_params)
        print("  ✅ Original TimeLockedTopography: Instantiated successfully")
    except Exception as e:
        print(
            f"  ❌ Original TimeLockedTopography: Failed to instantiate - {e}"
        )

    try:
        _ = OriginalTimeLockedContrast(**contrast_params)
        print("  ✅ Original TimeLockedContrast: Instantiated successfully")
    except Exception as e:
        print(f"  ❌ Original TimeLockedContrast: Failed to instantiate - {e}")

    # Test refactored markers
    try:
        _ = RefactoredTimeLockedTopography(**topo_params)
        print(
            "  ✅ Refactored TimeLockedTopography: Instantiated successfully"
        )
    except Exception as e:
        print(
            f"  ❌ Refactored TimeLockedTopography: Failed to instantiate - {e}"
        )

    try:
        _ = RefactoredTimeLockedContrast(**contrast_params)
        print("  ✅ Refactored TimeLockedContrast: Instantiated successfully")
    except Exception as e:
        print(
            f"  ❌ Refactored TimeLockedContrast: Failed to instantiate - {e}"
        )


def main():
    """Run registration verification tests."""
    print("=" * 70)
    print("MARKER REGISTRATION VERIFICATION")
    print("=" * 70)

    # Test registration
    registration_ok = test_marker_registration()

    # Test instantiation
    test_marker_instantiation()

    print("\n" + "=" * 70)
    if registration_ok:
        print("🎉 REGISTRATION VERIFICATION PASSED")
        print(
            "Refactored markers register correctly and can replace originals."
        )
    else:
        print("❌ REGISTRATION VERIFICATION FAILED")
        print(
            "Registration conflicts detected - need to resolve before deployment."
        )
    print("=" * 70)

    return registration_ok


if __name__ == "__main__":
    main()
