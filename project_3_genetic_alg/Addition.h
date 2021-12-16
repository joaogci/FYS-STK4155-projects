#pragma once

#include "Expression.h"

template<typename T>
class Addition : public Expression<T> {
private:
	ExpressionPtr<T> a;
	ExpressionPtr<T> b;
public:

	Addition(ExpressionPtr<T> a, ExpressionPtr<T> b) : a(a), b(b) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "(" + a->toString() + " + " + b->toString() + ")"; }

	std::string toJsString() const override { return "(" + a->toJsString() + " + " + b->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant() && b->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;
};
#define AdditionPtr(T, a, b) ExpressionPtr<T>(new Addition<T>(a, b))
#define AdditionPtrf(a, b) AdditionPtr(float, a, b)
#define AdditionPtrd(a, b) AdditionPtr(double, a, b)



template<typename T>
inline T Addition<T>::evaluate(T x, T y) const {
	return a->evaluate(x, y) + b->evaluate(x, y);
}

template<typename T>
inline ExpressionPtr<T> Addition<T>::derivative(int dimension) const {
	auto aPrime = a->derivative(dimension);
	auto bPrime = b->derivative(dimension);
	return AdditionPtr(T, aPrime, bPrime);
}

template<typename T>
inline ExpressionPtr<T> Addition<T>::simplify() const {
	ExpressionPtr<T> a_s = a->simplify();
	ExpressionPtr<T> b_s = b->simplify();
	bool constA = a_s->isConstant();
	bool constB = b_s->isConstant();
	if (constA && constB) {
		return ConstantPtr(T, a_s->evaluate(0, 0) + b_s->evaluate(0, 0));
	}
	if (constA && a_s->evaluate(0, 0) == 0) {
		return b_s;
	}
	if (constB && b_s->evaluate(0, 0) == 0) {
		return a_s;
	}
	return AdditionPtr(T, a_s, b_s);
}

template<typename T>
inline ExpressionPtr<T> Addition<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	auto newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	auto newB = b->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	if (MUTATION) {
		return grammar->instantiateOperation(newA, newB, rng);
	}
	return AdditionPtr(T, newA, newB);
}
