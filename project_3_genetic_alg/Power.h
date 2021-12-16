#pragma once

#include "Expression.h"
#include "Multiplication.h"

template<typename T>
class Power : public Expression<T> {
private:
	ExpressionPtr<T> a;
	ExpressionPtr<T> b;
public:

	Power(ExpressionPtr<T> a, ExpressionPtr<T> b) : a(a), b(b) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "(" + a->toString() + "^" + b->toString() + ")"; }

	std::string toJsString() const override { return "Math.pow(" + a->toJsString() + ", " + b->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;

};
#define PowerPtr(T, a, b) ExpressionPtr<T>(new Power<T>(a, b))
#define PowerPtrf(a, b) PowerPtr(float, a, b)
#define PowerPtrd(a, b) PowerPtr(double, a, b)



template<typename T>
inline T Power<T>::evaluate(T x, T y) const {
	T inner = a->evaluate(x, y);
	T outer = b->evaluate(x, y);
	if (inner <= 0) {
		throw NAN;
	}
	return pow(inner, outer);
}

template<typename T>
inline ExpressionPtr<T> Power<T>::derivative(int dimension) const {
	if (!b->isConstant()) { // we don't allow non-constant exponents
		return DivisionPtr(T, ConstantPtr(T, 1), ConstantPtr(T, 0)); // return 1/0 as the derivative, which will mark the expression as invalid if we evaluate it
	}
	// d/dx f(x)^c = c * f(x)^(c-1) * f'(x)
	T c = b->evaluate(0, 0);
	return MultiplicationPtr(T, MultiplicationPtr(T, ConstantPtr(T, c), a->derivative(dimension)), PowerPtr(T, a, ConstantPtr(T, c-1)));
}

template<typename T>
inline ExpressionPtr<T> Power<T>::simplify() const {
	if (a->isConstant() && b->isConstant()) {
		return ConstantPtr(T, pow(a->evaluate(0, 0), b->evaluate(0, 0)));
	}
	if (b->isConstant() && b->evaluate(0, 0) == 0) {
		return ConstantPtr(T, 1);
	}
	if (b->isConstant() && b->evaluate(0, 0) == 1) {
		return a->simplify();
	}
	return PowerPtr(T, a->simplify(), b->simplify());
}

template<typename T>
inline ExpressionPtr<T> Power<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	auto newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	auto newB = b->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	if (MUTATION) {
		return grammar->instantiateOperation(newA, newB, rng);
	}
	return PowerPtr(T, newA, newB);
}
