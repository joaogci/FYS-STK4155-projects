#pragma once

#include "Expression.h"
#include "Multiplication.h"
#include "Division.h"

template<typename T>
class SquareRoot : public Expression<T> {
private:
	ExpressionPtr<T> a;
public:

	SquareRoot(ExpressionPtr<T> a) : a(a) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "sqrt(" + a->toString() + ")"; }

	std::string toJsString() const override { return "Math.sqrt(" + a->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;

};
#define SquareRootPtr(T, a) ExpressionPtr<T>(new SquareRoot<T>(a))
#define SquareRootPtrf(a) SquareRootPtr(float, a)
#define SquareRootPtrd(a) SquareRootPtr(double, a)



template<typename T>
inline T SquareRoot<T>::evaluate(T x, T y) const {
	T inner = a->evaluate(x, y);
	if (inner <= 0) {
		throw NAN;
	}
	return sqrt(inner);
}

template<typename T>
inline ExpressionPtr<T> SquareRoot<T>::derivative(int dimension) const {
	// d/dx sqrt(f(x)) = f'(x) / (2sqrt(f(x)))
	return DivisionPtr(T, a->derivative(dimension), MultiplicationPtr(T, ConstantPtr(T, 2), SquareRootPtr(T, a)));
}

template<typename T>
inline ExpressionPtr<T> SquareRoot<T>::simplify() const {
	if (a->isConstant()) {
		return ConstantPtr(T, sqrt(a->evaluate(0, 0)));
	}
	return SquareRootPtr(T, a->simplify());
}

template<typename T>
inline ExpressionPtr<T> SquareRoot<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	auto newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	if (MUTATION) {
		return grammar->instantiateFunction(newA, rng);
	}
	return SquareRootPtr(T, newA);
}
