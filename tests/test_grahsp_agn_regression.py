"""Static regressions against GRAHSP AGN components.

Reference values were generated from GRAHSP commit
7d35f5232ac9918a785e8dfe75dff693ab246daf. GRAHSP evaluates spectral
densities per nm; the expected arrays below are converted to per Angstrom.
"""

from hashlib import sha256
from importlib import resources
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jaxsedfit.config import EmissionLineTemplate, FeIITemplate
from jaxsedfit.model import (
    _apply_biattenuation,
    _attenuation_curve,
    _balmer_continuum_jax,
    _line_gaussians,
    _powerlaw_jax,
    _torus_component,
)
from jaxsedfit.preload import _TEMPLATE_CACHE, _load_templates


GRAHSP_COMMIT = "7d35f5232ac9918a785e8dfe75dff693ab246daf"


def test_grahsp_agn_template_resources_match_upstream_commit():
    expected_hashes = {
        "resources/templates/Fe_d11-m20-20.5.txt": (
            "12bbb26587f60fa0d57bf0a4dbd30bf8cf9a537023ef3ed22f1b3592c3b8de4a"
        ),
        "resources/templates/emission_line_table.formatted": (
            "ac9b0a7bd9fa6236d220c4ac502f11676ac6504afc13efc8ec9062b03e48482a"
        ),
    }

    for relative_path, expected_hash in expected_hashes.items():
        payload = resources.files("jaxsedfit").joinpath(relative_path).read_bytes()
        assert sha256(payload).hexdigest() == expected_hash


def test_grahsp_default_feii_and_line_templates_match_static_reference():
    cfg = SimpleNamespace(
        agn=SimpleNamespace(
            fit_agn=True,
            feii_template=FeIITemplate(),
            emission_line_template=EmissionLineTemplate(),
        ),
        spectroscopy_config=SimpleNamespace(enabled=False, backend="jaxsedfit"),
        galaxy=SimpleNamespace(fit_host=False, use_energy_balance=False, dust_alpha=2.0),
    )
    _TEMPLATE_CACHE.clear()
    templates = _load_templates(cfg)

    feii_indices = np.asarray([0, 100, 500, 643])
    np.testing.assert_allclose(
        templates.feii_wave[feii_indices],
        [1993.123938694649, 2421.2719880698396, 5273.311490399269, 6965.1870074454655],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        templates.feii_lumin[feii_indices],
        [2.4058841096119687, 5.204779308595264, 0.27931095509872544, 0.0005717829844042825],
        rtol=2.0e-12,
        atol=0.0,
    )

    line_indices = np.asarray([0, 5, 10, 20, 30, 35])
    np.testing.assert_array_equal(
        templates.line_wave[line_indices],
        [1026.0, 1336.0, 1663.0, 4959.0, 8498.0, 18750.0],
    )
    np.testing.assert_array_equal(templates.line_blagn[line_indices], [1.0, 0.3, 0.5, 0.0, 0.1, 0.4])
    np.testing.assert_array_equal(templates.line_sy2[line_indices], [0.1, 0.0, 0.5, 3.0, 0.0, 0.3])
    np.testing.assert_array_equal(templates.line_liner[line_indices], [0.1, 0.0, 0.0, 0.8, 0.0, 0.3])


def test_agn_disk_matches_grahsp_activatepl_static_reference():
    wave_a = np.asarray([700.0, 1000.0, 2500.0, 5100.0, 20000.0, 100000.0])
    expected_per_a = np.asarray(
        [
            5.1161867868723385e33,
            3.700128726932793e33,
            1.5250419546215685e33,
            7.254901938650643e32,
            1.5282705050830885e32,
            1.2802504137871073e31,
        ]
    )

    disk = np.asarray(
        _powerlaw_jax(
            wave_a,
            norm=3.7e36 / 5100.0,
            lam1=0.0,
            lam2=-1.85,
            x0=5100.0,
            xbrk=1000.0,
            bend_width=10.0,
            cutoff=100000.0,
        )
    )

    np.testing.assert_allclose(disk, expected_per_a, rtol=2.0e-12, atol=0.0)


def test_torus_continuum_matches_grahsp_activategtorus_static_reference():
    # These are points on GRAHSP's internal torus grid, including 12 micron.
    wave_a = np.asarray([3600.0, 10000.0, 120000.0, 170600.0, 1000000.0])
    expected_per_a = np.asarray(
        [
            1.2286768482690989e31,
            7.862251437563254e31,
            2.6979166666666667e31,
            2.2510564328561865e31,
            1.020394917673146e30,
        ]
    )

    torus = np.asarray(
        _torus_component(
            wave_a,
            fcov=0.35,
            si=0.0,
            cool_lam=17.0,
            cool_width=0.45,
            hot_lam=2.0,
            hot_width=0.5,
            hot_fcov=0.7,
            si_ratio=0.29,
            si_em_lam=98410.0,
            si_abs_lam=142240.0,
            si_em_width=10253.0,
            si_abs_width=11635.0,
            l_agn=3.7e36,
        )
    )

    np.testing.assert_allclose(torus, expected_per_a, rtol=2.0e-12, atol=0.0)


def test_agn_line_profile_matches_grahsp_activatelines_static_reference():
    wave_a = np.asarray(
        [
            4849.8961571608315,
            4887.422117870624,
            4924.948078580415,
            4962.474039290208,
            5000.0,
            5037.525960709792,
            5075.051921419585,
            5112.577882129376,
            5150.1038428391685,
        ]
    )
    expected_per_a = np.asarray(
        [
            6.576456372366866e22,
            3.6242185182223296e27,
            8.826770325901406e30,
            9.500671392408042e32,
            4.5193064068618685e33,
            9.500671392408042e32,
            8.826770325901406e30,
            3.6242185182223296e27,
            6.576456372366866e22,
        ]
    )
    line_strength_per_a = 0.02 * (3.7e36 / 5100.0) * 2.3

    line = np.asarray(
        _line_gaussians(
            wave_a,
            np.asarray([5000.0]),
            np.asarray([line_strength_per_a]),
            3000.0,
        )
    )

    np.testing.assert_allclose(line, expected_per_a, rtol=2.0e-12, atol=0.0)


def test_biattenuation_matches_grahsp_static_reference():
    wave_a = np.asarray([1000.0, 5500.0, 11000.0, 22000.0, 400000.0])
    host = np.asarray([2.0, 3.0, 5.0, 7.0, 11.0])
    agn = np.asarray([13.0, 11.0, 7.0, 5.0, 3.0])
    expected_curve = np.asarray(
        [21.323204313868747, 2.7568760519928834, 1.2, 0.15, 2.4956249999999993e-5]
    )
    expected_host = np.asarray(
        [0.03937074660196369, 1.8053905735656048, 4.008390316938396, 6.809230566438756, 10.999949431893715]
    )
    expected_agn = np.asarray(
        [0.0002647514840837368, 2.721948811768377, 3.8115185698969483, 4.634149116896746, 2.9999620740728417]
    )

    curve = np.asarray(_attenuation_curve(wave_a, -1.2, -3.0, 1.2, 11000.0))
    host_att, agn_att, _, _ = _apply_biattenuation(
        wave_a,
        host,
        agn,
        ebv_gal=0.2,
        ebv_agn=0.35,
        opt_index=-1.2,
        nir_index=-3.0,
        norm=1.2,
        lam_break=11000.0,
    )

    np.testing.assert_allclose(curve, expected_curve, rtol=2.0e-12, atol=0.0)
    np.testing.assert_allclose(host_att, expected_host, rtol=2.0e-12, atol=0.0)
    np.testing.assert_allclose(agn_att, expected_agn, rtol=2.0e-12, atol=0.0)


def test_balmer_continuum_matches_grahsp_activatelines_static_reference():
    wave_a = np.asarray([2000.0, 2400.0, 2500.0, 3000.0, 3500.0, 3646.0, 3700.0])
    expected_per_a = np.asarray(
        [
            1.0402884492667541,
            1.532217526900752,
            1.6332469423985052,
            1.9829152680878477,
            2.0235488951859155,
            0.9856281353537738,
            0.0,
        ]
    )

    balmer = np.asarray(
        _balmer_continuum_jax(
            wave_a,
            balmer_norm=2.0,
            balmer_te=15000.0,
            balmer_tau=1.0,
            balmer_vel=3000.0,
        )
    )

    np.testing.assert_allclose(balmer, expected_per_a, rtol=2.0e-12, atol=1.0e-15)


def test_grahsp_agn_component_parameters_retain_finite_gradients():
    disk_grad = jax.grad(
        lambda slope: jnp.sum(
            _powerlaw_jax(
                jnp.asarray([1000.0, 2500.0, 5100.0, 20000.0]),
                1.0,
                0.0,
                slope,
                5100.0,
                1000.0,
                10.0,
                100000.0,
            )
        )
    )(-1.85)
    torus_grad = jax.grad(
        lambda fcov: jnp.sum(
            _torus_component(
                jnp.asarray([10000.0, 120000.0, 170600.0]),
                fcov,
                1.0,
                17.0,
                0.45,
                2.0,
                0.5,
                0.7,
                0.29,
                98410.0,
                142240.0,
                10253.0,
                11635.0,
                1.0,
            )
        )
    )(0.35)
    line_grad = jax.grad(
        lambda width: jnp.sum(
            _line_gaussians(
                jnp.asarray([4950.0, 5000.0, 5050.0]),
                jnp.asarray([5000.0]),
                jnp.asarray([1.0]),
                width,
            )
        )
    )(3000.0)
    attenuation_grad = jax.grad(
        lambda opt_index: jnp.sum(
            _attenuation_curve(
                jnp.asarray([1000.0, 5500.0, 22000.0]),
                opt_index,
                -3.0,
                1.2,
                11000.0,
            )
        )
    )(-1.2)
    balmer_grad = jax.grad(
        lambda tau: jnp.sum(
            _balmer_continuum_jax(
                jnp.asarray([2000.0, 2800.0, 3200.0, 3500.0]),
                1.0,
                15000.0,
                tau,
                3000.0,
            )
        )
    )(1.0)

    gradients = np.asarray([disk_grad, torus_grad, line_grad, attenuation_grad, balmer_grad])
    assert np.all(np.isfinite(gradients))
    assert np.all(gradients != 0.0)
