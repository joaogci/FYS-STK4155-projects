#pragma once

#include "Expression.h"
#include "Multiplication.h"

template<typename T>
class Exponential : public Expression<T> {
private:
	ExpressionPtr<T> a;
public:

	Exponential(ExpressionPtr<T> a) : a(a) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "exp(" + a->toString() + ")"; }

	std::string toJsString() const override { return "Math.exp(" + a->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;

};
#define ExponentialPtr(T, a) ExpressionPtr<T>(new Exponential<T>(a))
#define ExponentialPtrf(a) ExponentialPtr(float, a)
#define ExponentialPtrd(a) ExponentialPtr(double, a)



template<typename T>
inline T Exponential<T>::evaluate(T x, T y) const {
	return exp(a->evaluate(x, y));
}

template<typename T>
inline ExpressionPtr<T> Exponential<T>::derivative(int dimension) const {
	return MultiplicationPtr(T, a->derivative(dimension), ExponentialPtr(T, a));
}

template<typename T>
inline ExpressionPtr<T> Exponential<T>::simplify() const {
	if (a->isConstant()) {
		return ConstantPtr(T, exp(a->evaluate(0, 0)));
	}
	return ExponentialPtr(T, a->simplify());
}

template<typename T>
inline ExpressionPtr<T> Exponential<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	auto newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	if (MUTATION) {
		return grammar->instantiateFunction(newA, rng);
	}
	return ExponentialPtr(T, newA);
}
