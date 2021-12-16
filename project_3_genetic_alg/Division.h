#pragma once

#include "Expression.h"
#include "Multiplication.h"
#include "Subtraction.h"

template<typename T>
class Division : public Expression<T> {
private:
	ExpressionPtr<T> a;
	ExpressionPtr<T> b;
public:

	Division(ExpressionPtr<T> a, ExpressionPtr<T> b) : a(a), b(b) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "(" + a->toString() + " / " + b->toString() + ")"; }

	std::string toJsString() const override { return "(" + a->toJsString() + " / " + b->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant() && b->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;
};
#define DivisionPtr(T, a, b) ExpressionPtr<T>(new Division<T>(a, b))
#define DivisionPtrf(a, b) DivisionPtr(float, a, b)
#define DivisionPtrd(a, b) DivisionPtr(double, a, b)



template<typename T>
inline T Division<T>::evaluate(T x, T y) const {
	T denominator = b->evaluate(x, y);
	if (denominator == 0) {
		throw NAN;
	}
	return a->evaluate(x, y) / denominator;
}

template<typename T>
inline ExpressionPtr<T> Division<T>::derivative(int dimension) const {
	// f(x) = a(x) / b(x)
	// f' = (a'b - ab') / b^2
	auto aPrime = a->derivative(dimension);
	auto bPrime = b->derivative(dimension);
	auto aPrimeB = MultiplicationPtr(T, aPrime, b);
	auto bPrimeA = MultiplicationPtr(T, bPrime, a);
	return DivisionPtr(T, SubtractionPtr(T, aPrimeB, bPrimeA), MultiplicationPtr(T, b, b));
}

template<typename T>
inline ExpressionPtr<T> Division<T>::simplify() const {
	ExpressionPtr<T> a_s = a->simplify();
	ExpressionPtr<T> b_s = b->simplify();
	bool constA = a_s->isConstant();
	bool constB = b_s->isConstant();
	if (constA && constB) {
		return ConstantPtr(T, a_s->evaluate(0, 0) / b_s->evaluate(0, 0));
	}
	if (constA && a_s->evaluate(0, 0) == 0) {
		return ConstantPtr(T, 0);
	}
	if (constB && b_s->evaluate(0, 0) == 1) {
		return a_s;
	}
	return DivisionPtr(T, a_s, b_s);
}

template<typename T>
inline ExpressionPtr<T> Division<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	ExpressionPtr<T> newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	ExpressionPtr<T> newB = b->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	if (MUTATION) {
		return grammar->instantiateOperation(newA, newB, rng);
	}
	return DivisionPtr(T, newA, newB);
}
