#pragma once

#include "Fitness.h"

#define DOMAIN01 Domain<double>(0, 1, 50)

Fitness<double> pde1() {
	return Fitness<double>(
		[](FunctionParams<double> p) -> const double {
			return -p.ddx2-p.ddy2 + exp(-p.x) * (p.x - 2 + p.y*p.y*p.y + 6 * p.y);
		},
		DOMAIN01, DOMAIN01, 100,
		{
			Boundary<double>(0, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + y * y * y; // psi(0, y) = y^3
			}),
			Boundary<double>(1, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + (1+y*y*y) * exp(-1); // psi(1, y) = (1+y^3)exp(-1)
			}),
			Boundary<double>(0, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + x * exp(-x); // psi(x, 0) = x exp(-x)
			}),
			Boundary<double>(1, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + (x+1) * exp(-x); // psi(x, 1) = (x+1)exp(-x)
			})
		}
	);
}

Fitness<double> pde2() {
	return Fitness<double>(
		[](FunctionParams<double> p) -> const double {
			return -p.ddx2 - p.ddy2 - 2 * p.f;
		},
		DOMAIN01, DOMAIN01, 100,
		{
			Boundary<double>(0, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + 0; // psi(0, y) = 0
			}),
			Boundary<double>(1, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + sin(1)*cos(y); // psi(1, y) = sin(1)cos(y)
			}),
			Boundary<double>(0, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + sin(x); // psi(x, 0) = sin(x)
			}),
			Boundary<double>(1, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + sin(x) * cos(1); // psi(x, 1) = sin(x)cos(1)
			})
		}
	);
}

Fitness<double> pde3() {
	return Fitness<double>(
		[](FunctionParams<double> p) -> const double {
			return p.ddx2 + p.ddy2 - 4;
		},
		DOMAIN01, DOMAIN01, 100,
		{
			Boundary<double>(0, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + y * y + y + 1; // psi(0, y) = y^2 + y + 1
			}),
			Boundary<double>(1, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + y * y + y + 3; // psi(1, y) = y^2 + y + 3
			}),
			Boundary<double>(0, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + x * x + x + 1; // psi(x, 0) = x^2 + x + 1
			}),
			Boundary<double>(1, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + x * x + x + 3; // psi(x, 1) = x^2 + x + 3
			})
		}
	);
}

Fitness<double> pde4() {
	return Fitness<double>(
		[](FunctionParams<double> p) -> const double {
			return -p.ddx2 - p.ddy2 - (p.x*p.x + p.y*p.y) * p.f;
		},
		DOMAIN01, DOMAIN01, 100,
		{
			Boundary<double>(0, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + 0; // psi(0, y) = 0
			}),
			Boundary<double>(1, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + sin(y); // psi(1, y) = sin(y)
			}),
			Boundary<double>(0, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + 0; // psi(x, 0) = 0
			}),
			Boundary<double>(1, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + sin(x); // psi(x, 1) = sin(x)
			})
		}
	);
}

Fitness<double> pde5() {
	return Fitness<double>(
		[](FunctionParams<double> p) -> const double {
			return -p.ddx2 - p.ddy2 + (p.x - 2) * exp(-p.x) + p.x * exp(-p.y);
		},
		DOMAIN01, DOMAIN01, 100,
		{
			Boundary<double>(0, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + 0; // psi(0, y) = 0
			}),
			Boundary<double>(1, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + exp(-y) + exp(-1); // psi(1, y) = exp(-y) + exp(-1)
			}),
			Boundary<double>(0, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + x * (exp(-x)+1); // psi(x, 0) = x(exp(-x)+1)
			}),
			Boundary<double>(1, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + x * (exp(-x) + exp(-1)); // psi(x, 1) = x(exp(-x)+exp(-1))
			})
		}
	);
}

Fitness<double> pde6() {
	return Fitness<double>(
		[](FunctionParams<double> p) -> const double {
			return -p.ddx2 - p.ddy2 - exp(p.f) + 1 + p.x * p.x + p.y * p.y + 4 / ((1 + p.x * p.x + p.y * p.y) * (1 + p.x * p.x + p.y * p.y));
		},
		DOMAIN01, DOMAIN01, 100,
		{
			Boundary<double>(0, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + log(1 + y * y); // psi(0, y) = log(1+y^2)
			}),
			Boundary<double>(1, 0, [](double y, double f, double dfdx, double ddfdx) -> double {
				return -f + log(2 + y * y); // psi(1, y) = log(2+y^2)
			}),
			Boundary<double>(0, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + log(1+x*x); // psi(x, 0) = log(1+x^2)
			}),
			Boundary<double>(1, 1, [](double x, double f, double dfdy, double ddfdy) -> double {
				return -f + log(2+x*x); // psi(x, 1) = log(2+x^2)
			})
		}
	);
}



Fitness<double> getExamplePDE(int n) {
	switch (n) {
	case 2: return pde2();
	case 3: return pde3();
	case 4: return pde4();
	case 5: return pde5();
	case 6: return pde6();
	case 7: throw "Unimplemented: PDE7 (we do not support 3D PDEs at the moment, though it's an easy addition)";
	default: return pde1();
	}
}
