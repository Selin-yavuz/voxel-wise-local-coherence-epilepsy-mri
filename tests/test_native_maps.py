import numpy as np

from vic.native_maps import FeatureExtractionWithNativeMap, NativeMapParams, ReshapeMeta


def test_native_map_params_names_are_stable() -> None:
    params = NativeMapParams(
        sequence="t1_tra",
        shape="circle",
        primary_rate=3,
        secondary_rate=7,
        step=4,
        power=1,
        threshold=139.3,
    )

    assert params.model_id == "t1_tra|circle|PR3|SR7|STEP4|PW1"
    assert params.folder_name == "t1_tra_circle_PR3_SR7_PW1_STEP4"


def test_rate_map_to_native_space_inverts_min_slice_2_orientation() -> None:
    mapper = FeatureExtractionWithNativeMap.__new__(FeatureExtractionWithNativeMap)
    mapper.reshape_meta = ReshapeMeta(
        original_shape=(2, 3, 4),
        reoriented_shape=(4, 2, 3),
        crop_offsets=(1, 0, 0),
        min_slice=2,
    )

    rate_map = np.ones((2, 2, 3), dtype=np.float32)
    native = mapper.rate_map_to_native_space(rate_map)

    assert native.shape == (2, 3, 4)
    assert np.isfinite(native[:, :, 1]).all()
    assert np.isfinite(native[:, :, 2]).all()
    assert np.isnan(native[:, :, 0]).all()
    assert np.isnan(native[:, :, 3]).all()

