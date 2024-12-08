#!/usr/bin/env python

import pytest
import numpy as np
from operator import itemgetter

from gpr1d.core.kernels import (
    _Kernel,
    _OperatorKernel,
    _WarpingFunction,
    Constant_Kernel,
    Noise_Kernel,
    Linear_Kernel,
    Poly_Order_Kernel,
    SE_Kernel,
    RQ_Kernel,
    Matern_HI_Kernel,
    Gibbs_Kernel,
    Sum_Kernel,
    Product_Kernel,
    Constant_WarpingFunction,
    IG_WarpingFunction,
)
from gpr1d.core.routines import (
    GaussianProcessRegression1D,
)
from gpr1d.core.simple import (
    SimplifiedGaussianProcessRegression1D
)


@pytest.fixture(scope='module')
def empty_warping_function():
    return _WarpingFunction()

@pytest.fixture(scope='module')
def constant_warping_function():
    return Constant_WarpingFunction(1.0)

@pytest.fixture(scope='module')
def inverse_gaussian_warping_function():
    return IG_WarpingFunction(0.5, 0.4, 0.05, 0.9, 0.8)

@pytest.fixture(scope='module')
def empty_kernel():
    return _Kernel()

@pytest.fixture(scope='module')
def constant_kernel():
    return Constant_Kernel(2.0)

@pytest.fixture(scope='module')
def noise_kernel():
    return Noise_Kernel(1.0)

@pytest.fixture(scope='module')
def linear_kernel():
    return Linear_Kernel(2.0)

@pytest.fixture(scope='module')
def poly_order_kernel():
    return Poly_Order_Kernel(2.0, 1.0)

@pytest.fixture(scope='module')
def se_kernel():
    return SE_Kernel(1.0, 0.5)

@pytest.fixture(scope='module')
def rq_kernel():
    return RQ_Kernel(1.0, 0.5, 5.0)

@pytest.fixture(scope='module')
def matern_hi_kernel():
    return Matern_HI_Kernel(1.0, 0.5, 2.5)

@pytest.fixture(scope='module')
def gibbs_constant_kernel(constant_warping_function):
    return Gibbs_Kernel(1.0, constant_warping_function)

@pytest.fixture(scope='module')
def gibbs_inverse_gaussian_kernel(inverse_gaussian_warping_function):
    return Gibbs_Kernel(1.0, inverse_gaussian_warping_function)

@pytest.fixture(scope='module')
def empty_operator_kernel():
    return _OperatorKernel()

@pytest.fixture(scope='module')
def sum_kernel(se_kernel,noise_kernel):
    return Sum_Kernel(se_kernel, noise_kernel)

@pytest.fixture(scope='module')
def product_kernel(linear_kernel):
    return Product_Kernel(linear_kernel, linear_kernel)

@pytest.fixture(scope='module')
def linear_test_data():
    # Made to follow y = 1.9 * x with x_error = N(0,0.02) and y_error = N(0,0.2)
    xvalues = np.array([-0.96740013, -0.87351828, -0.82642837, -0.70047194, -0.59080932,
                        -0.52655153, -0.42021946, -0.30265796, -0.18695111, -0.06505908,
                         0.01570590,  0.10126372,  0.19727240,  0.28985247,  0.41903549,
                         0.54407216,  0.61490052,  0.67747007,  0.81998975,  0.90411010,
                         0.95085251])
    xerrors = np.full(xvalues.shape, 0.02)
    yvalues = np.array([-1.99033519, -1.77994779, -1.81395185, -1.34285693, -1.18439277,
                        -0.76244458, -0.69355271, -0.45994556, -0.22130082, -0.17259663,
                         0.18375909,  0.20151468,  0.20361261,  0.42478122,  0.97264555,
                         0.97186543,  1.15666269,  0.96138111,  1.69555386,  1.75694618,
                         1.77774039])
    yerrors = np.abs(yvalues) * 0.1
    gpinput = np.vstack((xvalues, yvalues, yerrors, xerrors))
    xoutput = np.linspace(-1.0, 1.0, 31)  # Chosen to be distinct from input x vector
    ref_mean = np.array([-1.92347366, -1.79560673, -1.66773979, -1.53987285, -1.41200592,
                         -1.28413898, -1.15627204, -1.02840511, -0.90053817, -0.77267124,
                         -0.64480430, -0.51693736, -0.38907043, -0.26120349, -0.13333656,
                         -0.00546962,  0.12239732,  0.25026425,  0.37813119,  0.50599812,
                          0.63386506,  0.76173200,  0.88959893,  1.01746587,  1.14533280,
                          1.27319974,  1.40106668,  1.52893361,  1.65680055,  1.78466749,
                          1.91253442])
    ref_std = np.array([0.12399159, 0.12306362, 0.12219331, 0.12138190, 0.12063059,
                        0.11994049, 0.11931268, 0.11874815, 0.11824779, 0.11781242,
                        0.11744278, 0.11713947, 0.11690303, 0.11673384, 0.11663221,
                        0.11659832, 0.11663221, 0.11673384, 0.11690303, 0.11713947,
                        0.11744278, 0.11781242, 0.11824779, 0.11874815, 0.11931268,
                        0.11994049, 0.12063059, 0.1213819 , 0.12219331, 0.12306362,
                        0.12399159])
    ref_derivative_mean = np.full(xoutput.shape, 1.91800404)
    ref_derivative_std = np.array([0.12399159, 0.13185388, 0.14099228, 0.15172738, 0.16449626,
                                   0.17991074, 0.19885447, 0.22265278, 0.25338812, 0.29453106,
                                   0.35232833, 0.43927302, 0.58451513, 0.87550381, 1.74948319,
                                   0.00491756, 1.74948319, 0.87550381, 0.58451513, 0.43927302,
                                   0.35232833, 0.29453106, 0.25338812, 0.22265278, 0.19885447,
                                   0.17991074, 0.16449626, 0.15172738, 0.14099228, 0.13185388,
                                   0.12399159])
    ref_target = np.vstack((ref_mean, ref_std, ref_derivative_mean, ref_derivative_std))
    ref_lml = 19.7879497328
    ref_null_lml = -1009.74859926
    hs_mean = np.array([-2.54429848, -2.37504322, -2.20578797, -2.03653271, -1.86727745,
                        -1.69802219, -1.52876694, -1.35951168, -1.19025642, -1.02100116,
                        -0.85174591, -0.68249065, -0.51323539, -0.34398013, -0.17472488,
                        -0.00546962,  0.16378564,  0.33304089,  0.50229615,  0.67155141,
                         0.84080667,  1.01006192,  1.17931718,  1.34857244,  1.51782770,
                         1.68708295,  1.85633821,  2.02559347,  2.19484873,  2.36410398,
                         2.53335924])
    hs_std = np.array([0.10600761, 0.10497150, 0.10399376, 0.10307605, 0.10222000,
                       0.10142716, 0.10069902, 0.10003700, 0.09944242, 0.09891650,
                       0.09846034, 0.09807490, 0.09776103, 0.09751941, 0.09735059,
                       0.09725494, 0.09723269, 0.09728387, 0.09740837, 0.09760592,
                       0.09787608, 0.09821823, 0.09863164, 0.09911540, 0.09966851,
                       0.1002898 , 0.10097803, 0.10173183, 0.10254975, 0.10343028,
                       0.10437183])
    hs_derivative_mean = np.full(xoutput.shape, 2.53882886)
    hs_derivative_std = np.array([0.10601130, 0.11247294, 0.11999606, 0.12884810, 0.13939371,
                                  0.15214330, 0.16783403, 0.18757147, 0.21309275, 0.24729284,
                                  0.29538233, 0.36778193, 0.48880593, 0.73139612, 1.46025912,
                                  0.00399690, 1.45849054, 0.72962954, 0.48704267, 0.36602328,
                                  0.29362956, 0.24554717, 0.21135536, 0.18584349, 0.16611654,
                                  0.15043731, 0.13770015, 0.12716786, 0.11832994, 0.11082169,
                                  0.10437558])
    hs_mean = np.array([-1.75502029, -1.63838358, -1.52174687, -1.40511016, -1.28847344,
                        -1.17183673, -1.05520002, -0.93856331, -0.82192660, -0.70528989,
                        -0.58865318, -0.47201647, -0.35537975, -0.23874304, -0.12210633,
                        -0.00546962,  0.11116709,  0.22780380,  0.34444051,  0.46107723,
                         0.57771394,  0.69435065,  0.81098736,  0.92762407,  1.04426078,
                         1.16089749,  1.27753420,  1.39417092,  1.51080763,  1.62744434,
                         1.74408105])
    hs_std = np.array([0.19877810, 0.19131875, 0.18076892, 0.16740661, 0.15171693,
                       0.13435209, 0.11607357, 0.09768536, 0.07996797, 0.06362132,
                       0.04922185, 0.03719763, 0.02782831, 0.02128965, 0.01775714,
                       0.01743202, 0.02027910, 0.02592881, 0.03396462, 0.04404844,
                       0.05587607, 0.06913114, 0.08346096, 0.09846393, 0.11368536,
                       0.12862426, 0.14275374, 0.1555547 , 0.16655913, 0.17539552,
                       0.18182756])
    hs_derivative_mean = np.full(xoutput.shape, 1.74955067)
    hs_derivative_std = np.array([0.21496683, 0.24285578, 0.27287635, 0.30052605, 0.32193352,
                                  0.33428406, 0.33598285, 0.32666075, 0.30707488, 0.27897202,
                                  0.24504937, 0.20935282, 0.17925773, 0.17366077, 0.26740397,
                                  0.01953941, 0.31007777, 0.21753406, 0.21446882, 0.23012135,
                                  0.24948387, 0.26747676, 0.28171492, 0.29072559, 0.29349435,
                                  0.28944770, 0.27859694, 0.26171101, 0.24042717, 0.21722140,
                                  0.19514516])
    hs_target = np.vstack((hs_mean, hs_std, hs_derivative_mean, hs_derivative_std))
    hs_lml = -46.6833105933
    hs_null_lml = -1007.28701166
    ni_mean = np.array([-1.74042085, -1.62475743, -1.50909402, -1.39343060, -1.27776719,
                        -1.16210377, -1.04644036, -0.93077694, -0.81511353, -0.69945011,
                        -0.58378670, -0.46812328, -0.35245987, -0.23679645, -0.12113304,
                        -0.00546962,  0.11019380,  0.22585721,  0.34152063,  0.45718404,
                         0.57284746,  0.68851087,  0.80417429,  0.91983770,  1.03550112,
                         1.15116453,  1.26682795,  1.38249136,  1.49815478,  1.61381819,
                         1.72948161])
    ni_std = np.array([0.20602120, 0.19744122, 0.18585382, 0.17167832, 0.15554576,
                       0.13823194, 0.12057128, 0.10336747, 0.08731798, 0.07296574,
                       0.06068516, 0.05070205, 0.04313957, 0.03807182, 0.03555171,
                       0.03558297, 0.03806477, 0.04278751, 0.04948947, 0.05791109,
                       0.06781651, 0.07899325, 0.09124076, 0.10434916, 0.11806966,
                       0.13208448, 0.14598845, 0.15929423, 0.17146629, 0.18197941,
                       0.19038861])
    ni_derivative_mean = np.full(xoutput.shape, 1.73495123)
    ni_derivative_std = np.array([0.22791510, 0.25689235, 0.28605774, 0.31068007, 0.32710750,
                                  0.33321264, 0.32853134, 0.31417169, 0.29257742, 0.26731469,
                                  0.24319884, 0.22736976, 0.23273144, 0.28997513, 0.53348137,
                                  0.01932138, 0.57326806, 0.33092916, 0.2698466 , 0.25388830,
                                  0.25421704, 0.26083285, 0.26939502, 0.27752553, 0.28345370,
                                  0.28556772, 0.28245637, 0.27322900, 0.2579224 , 0.23780856,
                                  0.21544209])
    ni_target = np.vstack((ni_mean, ni_std, ni_derivative_mean, ni_derivative_std))
    ni_lml = 3.98229778152
    ni_null_lml = -732.662302649
    xy_mean = np.array([-1.79799505, -1.67849335, -1.55899166, -1.43948996, -1.31998827,
                        -1.20048657, -1.08098488, -0.96148318, -0.84198149, -0.72247979,
                        -0.60297810, -0.4834764 , -0.36397471, -0.24447301, -0.12497132,
                        -0.00546962,  0.11403208,  0.23353377,  0.35303547,  0.47253716,
                         0.59203886,  0.71154055,  0.83104225,  0.95054394,  1.07004564,
                         1.18954733,  1.30904903,  1.42855072,  1.54805242,  1.66755411,
                         1.78705581])
    xy_std = np.array([0.20133217, 0.19379133, 0.18338442, 0.17041874, 0.15540358,
                       0.13900484, 0.12198051, 0.10510678, 0.08910691, 0.07459357,
                       0.06203402, 0.05174407, 0.04391171, 0.03863968, 0.03597620,
                       0.03590099, 0.03829011, 0.04293075, 0.04958616, 0.05804201,
                       0.06810521, 0.07957447, 0.09220527, 0.10568267, 0.11960885,
                       0.13350941, 0.14685916, 0.15912419, 0.16981368, 0.17853179,
                       0.18502017])
    xy_derivative_mean = np.full(xoutput.shape, 1.79252543)
    xy_derivative_std = np.array([0.21761201, 0.24396712, 0.27168392, 0.29653606, 0.31494765,
                                  0.32446884, 0.32399891, 0.31384643, 0.29569408, 0.27258904,
                                  0.24921672, 0.23302527, 0.23761033, 0.29459353, 0.53989551,
                                  0.01787825, 0.57650630, 0.33165946, 0.27009775, 0.25488990,
                                  0.25686483, 0.26532472, 0.27498041, 0.28257315, 0.28586099,
                                  0.28339810, 0.27459532, 0.25983562, 0.24052033, 0.21896468,
                                  0.19806665])
    xy_target = np.vstack((xy_mean, xy_std, xy_derivative_mean, xy_derivative_std))
    xy_lml = 4.83962150611
    xy_null_lml = -749.992013630
    return (
        gpinput, xoutput,
        ref_target, ref_lml, ref_null_lml,
        hs_target, hs_lml, hs_null_lml,
        ni_target, ni_lml, ni_null_lml,
        xy_target, xy_lml, xy_null_lml
    )

@pytest.fixture(scope='module')
def empty_gpr_object():
    return GaussianProcessRegression1D()

@pytest.fixture(scope='class')
def unoptimized_gpr_object(linear_kernel, linear_test_data):
    gpr_input = itemgetter(0)(linear_test_data)
    gpr_object = GaussianProcessRegression1D()
    gpr_object.set_kernel(kernel=linear_kernel)
    gpr_object.set_raw_data(xdata=gpr_input[0, :], ydata=gpr_input[1, :], yerr=gpr_input[2, :], xerr=gpr_input[3, :])
    return gpr_object

@pytest.fixture(scope='class')
def simplified_unoptimized_gpr_object(linear_kernel, linear_test_data):
    gpr_input = itemgetter(0)(linear_test_data)
    gpr_object = SimplifiedGaussianProcessRegression1D(
        kernel=linear_kernel,
        xdata=gpr_input[0, :],
        ydata=gpr_input[1, :],
        yerr=gpr_input[2, :],
        xerr=gpr_input[3, :],
        reg_par=1.0,
        epsilon='None'
    )
    gpr_object.set_error_search_parameters(epsilon='None')
    return gpr_object

@pytest.fixture(scope='module')
def gaussian_test_data():
    # Made to follow y = exp(- x^2 / (2 * 0.3^2)) with x_error = N(0,0.02) and y_error = N(0,0.05)
    xvalues = np.array([-1.00540703, -0.89343136, -0.81342899, -0.71212131, -0.61382335,
                        -0.47974300, -0.41572200, -0.31427158, -0.21452901, -0.08541332,
                        -0.04283059,  0.10003251,  0.20386169,  0.29962928,  0.40300245,
                         0.48250891,  0.63064336,  0.69389597,  0.80410067,  0.94612883,
                         0.99258774])
    xerrors = np.full(xvalues.shape, 0.02)
    yvalues = np.array([ 0.03085014,  0.07193564, -0.01577018,  0.07841518,  0.05814738,
                         0.34168284,  0.35344264,  0.5136707 ,  0.78512886,  0.91773780,
                         1.03413131,  0.89498977,  0.76891424,  0.67849064,  0.46100151,
                         0.2735394 ,  0.13281691,  0.07783454,  0.07562198,  0.04154204,
                         0.03022913])
    yerrors = np.full(yvalues.shape, 0.05)
    return (xvalues, xerrors, yvalues, yerrors)

@pytest.fixture(scope='function')
def preoptimization_gpr_object(rq_kernel, gaussian_test_data):
    xvalues, xerrors, yvalues, yerrors = gaussian_test_data
    gpr_object = GaussianProcessRegression1D()
    gpr_object.set_kernel(kernel=rq_kernel)
    gpr_object.set_raw_data(xdata=xvalues, ydata=yvalues, yerr=yerrors, xerr=xerrors)
    return gpr_object

