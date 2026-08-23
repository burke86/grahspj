from collections.abc import Mapping

import numpy as np

from jaxsedfit.results import PredictionResult


def test_prediction_result_satisfies_mapping_contract():
    data = {
        "pred_fluxes": np.array([[1.0], [3.0]]),
        "sed_chi2": np.array([1.0, 9.0]),
    }
    prediction = PredictionResult(data, fitter=None)

    assert isinstance(prediction, Mapping)
    assert "pred_fluxes" in prediction
    assert "missing" not in prediction
    assert list(prediction) == list(data)
    assert len(prediction) == len(data)
    assert prediction.keys() == data.keys()
    assert prediction.get("missing") is None
    np.testing.assert_array_equal(prediction["sed_chi2"], data["sed_chi2"])
    np.testing.assert_array_equal(prediction.median["pred_fluxes"], [2.0])
