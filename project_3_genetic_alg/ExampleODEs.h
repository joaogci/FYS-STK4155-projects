#pragma once

#include "Fitness.h"


/// Example ODEs and NLODEs from the original paper

#define ODE(deriv, function) \
	[](FunctionParams<double> p) -> const double { \
		double x = p.x; double y = p.f; double dy = p.ddx; double ddy = p.ddx2; \
		return -deriv + (function); \
	}

#define BOUNDARY_F(x_0, f_0) \
			Boundary<double>(x_0, 0, [](double y, double f, double df, double ddf) -> double { \
				return -f + f_0; \
			})
#define BOUNDARY_DFDX(x_0, df_0) \
			Boundary<double>(x_0, 0, [](double y, double f, double df, double ddf) -> double { \
				return -df + df_0; \
			})

Fitness<double> ode1() {
	return Fitness<double>(
		ODE(dy, (2 * x - y) / x),
		Domain<double>(0.1), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.1, 20.1)
		}
	);
}

Fitness<double> ode2() {
	return Fitness<double>(
		ODE(dy, (1-y*cos(x))/sin(x)),
		Domain<double>(0.1), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.1, 2.1/sin(0.1))
		}
	);
}

Fitness<double> ode3() {
	return Fitness<double>(
		ODE(dy, -y/5 + exp(-x/5) * cos(x)),
		Domain<double>(), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.0, 0.0)
		}
	);
}

Fitness<double> ode4() {
	return Fitness<double>(
		ODE(ddy, -100*y),
		Domain<double>(), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.0, 0.0),
			BOUNDARY_DFDX(0.0, 10.0)
		}
	);
}

Fitness<double> ode5() {
	return Fitness<double>(
		ODE(ddy, 6 * dy - 9 * y),
		Domain<double>(), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.0, 0.0),
			BOUNDARY_DFDX(0.0, 2.0)
		}
	);
}

Fitness<double> ode6() {
	return Fitness<double>(
		ODE(ddy, -dy / 5 - y - exp(-x / 5) * cos(x) / 5),
		Domain<double>(0, 2), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.0, 0.0),
			BOUNDARY_DFDX(0.0, 1.0)
		}
	);
}

Fitness<double> ode7() {
	return Fitness<double>(
		ODE(ddy, -100 * y),
		Domain<double>(), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.0, 0.0),
			BOUNDARY_F(1.0, sin(10.0))
		}
	);
}

Fitness<double> ode8() {
	return Fitness<double>(
		ODE(0, x * ddy + (1 - x) * dy + y),
		Domain<double>(0), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.0, 1.0),
			BOUNDARY_F(1.0, 0.0)
		}
	);
}

Fitness<double> ode9() {
	return Fitness<double>(
		ODE(ddy, -dy / 5 - y - exp(-x / 5) * cos(x) / 5),
		Domain<double>(0), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(0.0, 0.0),
			BOUNDARY_F(1.0, sin(1.0) / exp(0.2))
		}
	);
}

Fitness<double> nlode1() {
	return Fitness<double>(
		ODE(dy, 1 / (2 * y)),
		Domain<double>(1, 4), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(1.0, 1.0)
		}
	);
}

Fitness<double> nlode2() {
	return Fitness<double>(
		ODE(0, dy*dy + log(y) - cos(x)*cos(x) - 2*cos(x) - 1 - log(x+sin(x))),
		Domain<double>(1, 2), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(1.0, 1.0 + sin(1.0))
		}
	);
}

Fitness<double> nlode3() {
	return Fitness<double>(
		ODE(ddy * dy, -4/(x*x*x)),
		Domain<double>(1, 2), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(1.0, 0.0)
		}
	);
}

Fitness<double> nlode4() {
	return Fitness<double>(
		ODE(0, x*x*ddy + (x*dy)*(x*dy) + 1/log(x)),
		Domain<double>(exp(1), 2*exp(1)), Domain<double>(EMPTY), 100,
		{
			BOUNDARY_F(exp(1.0), 0.0),
			BOUNDARY_DFDX(exp(1.0), exp(-1.0))
		}
	);
}

Fitness<double> getExampleODE(int n) {
	switch (n) {
	case 2: return ode2();
	case 3: return ode3();
	case 4: return ode4();
	case 5: return ode5();
	case 6: return ode6();
	case 7: return ode7();
	case 8: return ode8();
	case 9: return ode9();
	default: return ode1();
	}
}

Fitness<double> getExampleNLODE(int n) {
	switch (n) {
	case 2: return nlode2();
	case 3: return nlode3();
	case 4: return nlode4();
	default: return nlode1();
	}
}

#undef ODE
#undef BOUNDARY_F
#undef BOUNDARY_DFDX
