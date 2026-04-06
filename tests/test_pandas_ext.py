"""Tests for pandas ExtensionDtype/ExtensionArray for Universal types."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import mtl5  # noqa: E402


@pytest.fixture
def Posit16Dtype():
    return mtl5.Posit16Dtype


@pytest.fixture
def Posit16Array():
    return mtl5.Posit16Array


class TestPosit16Dtype:
    def test_dtype_name(self, Posit16Dtype):
        assert Posit16Dtype().name == "posit16"

    def test_dtype_kind(self, Posit16Dtype):
        assert Posit16Dtype().kind == "f"

    def test_construct_from_string(self, Posit16Dtype):
        dt = Posit16Dtype.construct_from_string("posit16")
        assert isinstance(dt, Posit16Dtype)

    def test_construct_from_string_invalid(self, Posit16Dtype):
        with pytest.raises(TypeError):
            Posit16Dtype.construct_from_string("not_a_real_type")

    def test_repr(self, Posit16Dtype):
        assert repr(Posit16Dtype()) == "Posit16Dtype()"


class TestPosit16Array:
    def test_create_from_list(self, Posit16Array):
        arr = Posit16Array([1.0, 2.0, 3.0])
        assert len(arr) == 3
        assert arr[0] == pytest.approx(1.0, rel=1e-3)

    def test_create_from_numpy(self, Posit16Array):
        arr = Posit16Array(np.array([1.0, 2.0, 3.0]))
        assert len(arr) == 3

    def test_dtype_property(self, Posit16Array, Posit16Dtype):
        arr = Posit16Array([1.0, 2.0])
        assert isinstance(arr.dtype, Posit16Dtype)

    def test_nbytes(self, Posit16Array):
        arr = Posit16Array([1.0, 2.0, 3.0])
        assert arr.nbytes == 6  # 16 bits per element × 3

    def test_getitem_scalar(self, Posit16Array):
        arr = Posit16Array([1.0, 2.0, 3.0])
        assert arr[0] == pytest.approx(1.0, rel=1e-3)
        assert arr[2] == pytest.approx(3.0, rel=1e-3)

    def test_getitem_slice(self, Posit16Array):
        arr = Posit16Array([1.0, 2.0, 3.0, 4.0])
        sub = arr[1:3]
        assert isinstance(sub, Posit16Array)
        assert len(sub) == 2

    def test_setitem(self, Posit16Array):
        arr = Posit16Array([1.0, 2.0, 3.0])
        arr[1] = 99.0
        assert arr[1] == pytest.approx(99.0, rel=1e-2)

    def test_isna(self, Posit16Array):
        arr = Posit16Array([1.0, 2.0, 3.0])
        assert not arr.isna().any()

    def test_copy_independence(self, Posit16Array):
        arr = Posit16Array([1.0, 2.0, 3.0])
        copy = arr.copy()
        copy[0] = 99.0
        assert arr[0] == pytest.approx(1.0, rel=1e-3)

    def test_concat(self, Posit16Array):
        a = Posit16Array([1.0, 2.0])
        b = Posit16Array([3.0, 4.0])
        combined = Posit16Array._concat_same_type([a, b])
        assert len(combined) == 4

    def test_arithmetic_add(self, Posit16Array):
        a = Posit16Array([1.0, 2.0, 3.0])
        b = Posit16Array([4.0, 5.0, 6.0])
        result = a + b
        assert isinstance(result, Posit16Array)
        assert result[0] == pytest.approx(5.0, rel=1e-2)

    def test_arithmetic_scalar(self, Posit16Array):
        a = Posit16Array([1.0, 2.0, 3.0])
        result = a * 2.0
        assert isinstance(result, Posit16Array)
        assert result[1] == pytest.approx(4.0, rel=1e-2)

    def test_to_numpy(self, Posit16Array):
        arr = Posit16Array([1.0, 2.0, 3.0])
        np_arr = arr.to_numpy()
        assert np_arr.dtype == np.float64
        assert len(np_arr) == 3


class TestPandasSeriesIntegration:
    def test_create_series(self, Posit16Dtype):
        s = pd.Series([1.0, 2.0, 3.0], dtype=Posit16Dtype())
        assert isinstance(s.dtype, Posit16Dtype)
        assert len(s) == 3

    def test_series_sum(self, Posit16Dtype):
        s = pd.Series([1.0, 2.0, 3.0], dtype=Posit16Dtype())
        # sum() on extension arrays works through to_numpy
        total = float(np.asarray(s.array).sum())
        assert total == pytest.approx(6.0, rel=1e-3)

    def test_series_indexing(self, Posit16Dtype):
        s = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=Posit16Dtype())
        # Slicing returns a Series
        sub = s.iloc[1:3]
        assert len(sub) == 2

    def test_series_to_numpy(self, Posit16Dtype):
        s = pd.Series([1.0, 2.0, 3.0], dtype=Posit16Dtype())
        np_arr = s.to_numpy()
        assert np_arr.dtype == np.float64

    def test_construct_from_string_dtype(self):
        # The dtype should be discoverable by name
        s = pd.Series([1.0, 2.0, 3.0], dtype="posit16")
        assert s.dtype.name == "posit16"
