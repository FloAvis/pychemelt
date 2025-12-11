import numpy as np

from extra_tests.test import low_bounds
from pychemelt.utils.fitting import evaluate_need_to_refit

# Model / ground-truth parameters
DHm_VAL = 120
Tm_VAL = 65
CP0_VAL = 1.8
M0_VAL = 2.6
M1_VAL = 0

INTERCEPT_N = 100
PRE_EXP_N = 1
C_N_VAL = 0
ALPHA_N_VAL = 0.1
INTERCEPT_U = 110
PRE_EXP_U = 1
C_U_VAL = 0
ALPHA_U_VAL = 0.2

PARAMS = [Tm_VAL,DHm_VAL,CP0_VAL,M0_VAL,M1_VAL,INTERCEPT_N,INTERCEPT_U,C_N_VAL,C_U_VAL,ALPHA_N_VAL,ALPHA_U_VAL,PRE_EXP_N,PRE_EXP_U]
LOW_BOUNDS = [-np.inf for _ in PARAMS]
UPP_BOUNDS = [np.inf for _ in PARAMS]

def test_evaluate_need_to_refit():

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        UPP_BOUNDS,
        LOW_BOUNDS,
        PARAMS,
        fit_m1=True,
        check_cp=True,
        check_dh=True,
        check_tm=True,
        fixed_cp=False
    )

    assert p0 == PARAMS and not re_fit

def test_evaluate_need_to_refit_tm():

    upp_bounds_tm = UPP_BOUNDS.copy()
    upp_bounds_tm[0] = 70

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        upp_bounds_tm,
        LOW_BOUNDS,
        PARAMS,
        fit_m1=True,
        check_cp=False,
        check_dh=False,
        check_tm=True,
        fixed_cp=False
    )

    assert high_bounds[0] == (Tm_VAL + 12) and re_fit

    low_bounds_tm = LOW_BOUNDS.copy()
    low_bounds_tm[0] = 60

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        UPP_BOUNDS,
        low_bounds_tm,
        PARAMS,
        fit_m1=True,
        check_cp=False,
        check_dh=False,
        check_tm=True,
        fixed_cp=False
    )

    assert low_bounds[0] == (Tm_VAL - 12) and re_fit

def test_evaluate_need_to_refit_dh():

    upp_bounds_dh = UPP_BOUNDS.copy()
    upp_bounds_dh[1] = 130

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        upp_bounds_dh,
        LOW_BOUNDS,
        PARAMS,
        fit_m1=True,
        check_cp=False,
        check_dh=True,
        check_tm=False,
        fixed_cp=False
    )

    assert high_bounds[1] == (DHm_VAL + 80) and re_fit

def test_evaluate_need_to_refit_cp():

    upp_bounds_dh = UPP_BOUNDS.copy()
    upp_bounds_dh[2] = 1.9

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        upp_bounds_dh,
        LOW_BOUNDS,
        PARAMS,
        fit_m1=True,
        check_cp=True,
        check_dh=False,
        check_tm=False,
        fixed_cp=False
    )

    assert high_bounds[2] == (CP0_VAL + 1) and re_fit

def test_evaluate_need_to_fixed_cp():

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        UPP_BOUNDS,
        LOW_BOUNDS,
        PARAMS,
        fit_m1=True,
        check_cp=True,
        check_dh=False,
        check_tm=False,
        fixed_cp=True
    )

    assert not re_fit

def test_evaluate_need_to_refit_m0():

    upp_bounds_m0 = UPP_BOUNDS.copy()
    upp_bounds_m0[3] = 3

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        upp_bounds_m0,
        LOW_BOUNDS,
        PARAMS,
        fit_m1=True,
        check_cp=False,
        check_dh=False,
        check_tm=False,
        fixed_cp=False
    )

    assert high_bounds[3] == (M0_VAL + 2) and re_fit


def test_evaluate_need_to_refit_m1():

    upp_bounds_m1 = UPP_BOUNDS.copy()
    upp_bounds_m1[4] = 0.01

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        upp_bounds_m1,
        LOW_BOUNDS,
        PARAMS,
        fit_m1=True,
        check_cp=False,
        check_dh=False,
        check_tm=False,
        fixed_cp=False
    )

    assert high_bounds[4] == (M1_VAL + 1) and re_fit

    low_bounds_m1 = LOW_BOUNDS.copy()
    low_bounds_m1[4] = -0.01

    re_fit, p0, low_bounds, high_bounds = evaluate_need_to_refit(
        PARAMS,
        UPP_BOUNDS,
        low_bounds_m1,
        PARAMS,
        fit_m1=True,
        check_cp=False,
        check_dh=False,
        check_tm=False,
        fixed_cp=False
    )

    assert low_bounds[4] == (M1_VAL - 1) and re_fit

def test_evaluate_need_to_refit_other_params():

    upp_bounds_in = UPP_BOUNDS.copy()
    upp_bounds_in[6:] = [x + 0.1 for x in PARAMS[6:]]

    low_bounds_in = LOW_BOUNDS.copy()
    low_bounds_in[6:] = [x - 0.1 for x in PARAMS[6:]]

    re_fit, p0, low_bounds_out, upp_bounds_out = evaluate_need_to_refit(
        PARAMS,
        upp_bounds_in,
        low_bounds_in,
        PARAMS,
        fit_m1=True,
        check_cp=False,
        check_dh=False,
        check_tm=False,
        fixed_cp=False
    )

    assert re_fit

    for i in range(6,len(PARAMS)):

        assert low_bounds_out[i] == low_bounds_in[i] - 50
        assert upp_bounds_out[i] == upp_bounds_in[i] + 50

