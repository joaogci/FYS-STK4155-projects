#pragma once

#include "Expression.h"

template<typename T>
class Multiplication : public Expression<T> {
private:
	ExpressionPtr<T> a;
	ExpressionPtr<T> b;
public:

	Multiplication(ExpressionPtr<T> a, ExpressionPtr<T> b) : a(a), b(b) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "(" + a->toString() + " * " + b->toString() + ")"; }

	std::string toJsString() const override { return "(" + a->toJsString() + " * " + b->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant() && b->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;
};
#define MultiplicationPtr(T, a, b) ExpressionPtr<T>(new Multiplication<T>(a, b))
#define MultiplicationPtrf(a, b) MultiplicationPtr(float, a, b)
#define MultiplicationPtrd(a, b) MultiplicationPtr(double, a, b)



template<typename T>
inline T Multiplication<T>::evaluate(T x, T y) const {
	return a->evaluate(x, y) * b->evaluate(x, y);
}

template<typename T>
inline ExpressionPtr<T> Multiplication<T>::derivative(int dimension) const {
	auto aPrime = a->derivative(dimension);
	auto bPrime = b->derivative(dimension);
	auto aTimesBPrime = MultiplicationPtr(T, a, bPrime);
	auto bTimesAPrime = MultiplicationPtr(T, b, aPrime);
	return AdditionPtr(T, aTimesBPrime, bTimesAPrime);
}

template<typename T>
inline ExpressionPtr<T> Multiplication<T>::simplify() const {
	ExpressionPtr<T> a_s = a->simplify();
	ExpressionPtr<T> b_s = b->simplify();
	bool constA = a_s->isConstant();
	bool constB = b_s->isConstant();
	if (constA && constB) {
		return ConstantPtr(T, a_s->evaluate(0, 0) * b_s->evaluate(0, 0));
	}
	if ((constA && a_s->evaluate(0, 0) == 0) || (constB && b_s->evaluate(0, 0) == 0)) {
		return ConstantPtr(T, 0);
	}
	if (constA && a_s->evaluate(0, 0) == 1) {
		return b_s;
	}
	if (constB && b_s->evaluate(0, 0) == 1) {
		return a_s;
	}
	return MultiplicationPtr(T, a_s, b_s);
}

template<typename T>
inline ExpressionPtr<T> Multiplication<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	// prevent the first multiplication at the top of the tree from mutating
	if (!first) {
		TREE_MUTATION();
	}
	auto newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	auto newB = b->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	if (!first && MUTATION) {
		return grammar->instantiateOperation(newA, newB, rng);
	}
	return MultiplicationPtr(T, newA, newB);
}
