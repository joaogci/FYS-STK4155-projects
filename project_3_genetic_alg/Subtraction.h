#pragma once

#include "Expression.h"
#include "Multiplication.h"

template<typename T>
class Subtraction : public Expression<T> {
private:
	ExpressionPtr<T> a;
	ExpressionPtr<T> b;
public:

	Subtraction(ExpressionPtr<T> a, ExpressionPtr<T> b) : a(a), b(b) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "(" + a->toString() + " - " + b->toString() + ")"; }

	std::string toJsString() const override { return "(" + a->toJsString() + " - " + b->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant() && b->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;
};
#define SubtractionPtr(T, a, b) ExpressionPtr<T>(new Subtraction<T>(a, b))
#define SubtractionPtrf(a, b) SubtractionPtr(float, a, b)
#define SubtractionPtrd(a, b) SubtractionPtr(double, a, b)



template<typename T>
inline T Subtraction<T>::evaluate(T x, T y) const {
	return a->evaluate(x, y) - b->evaluate(x, y);
}

template<typename T>
inline ExpressionPtr<T> Subtraction<T>::derivative(int dimension) const {
	auto aPrime = a->derivative(dimension);
	auto bPrime = b->derivative(dimension);
	return SubtractionPtr(T, aPrime, bPrime);
}

template<typename T>
inline ExpressionPtr<T> Subtraction<T>::simplify() const {
	ExpressionPtr<T> a_s = a->simplify();
	ExpressionPtr<T> b_s = b->simplify();
	bool constA = a_s->isConstant();
	bool constB = b_s->isConstant();
	if (constA && constB) {
		return ConstantPtr(T, a_s->evaluate(0, 0) - b_s->evaluate(0, 0));
	}
	if (constA && a_s->evaluate(0, 0) == 0) {
		return MultiplicationPtr(T, ConstantPtr(T, -1), b_s); // if 0 - b, simplify to -1 * b
	}
	if (constB && b_s->evaluate(0, 0) == 0) {
		return a_s;
	}
	return SubtractionPtr(T, a_s, b_s);
}

template<typename T>
inline ExpressionPtr<T> Subtraction<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	auto newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, first);
	auto newB = b->mutate(rng, mutationChance, treeMutationChance, grammar, first);
	if (MUTATION) {
		return grammar->instantiateOperation(newA, newB, rng);
	}
	return SubtractionPtr(T, newA, newB);
}
