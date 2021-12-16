#pragma once

#include "Expression.h"
#include "Multiplication.h"

template<typename T>
class Sine : public Expression<T> {
private:
	ExpressionPtr<T> a;
public:

	Sine(ExpressionPtr<T> a) : a(a) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "sin(" + a->toString() + ")"; }

	std::string toJsString() const override { return "Math.sin(" + a->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;

};
#define SinePtr(T, a) ExpressionPtr<T>(new Sine<T>(a))
#define SinePtrf(a) SinePtr(float, a)
#define SinePtrd(a) SinePtr(double, a)

template<typename T>
class Cosine : public Expression<T> {
private:
	ExpressionPtr<T> a;
public:

	Cosine(ExpressionPtr<T> a) : a(a) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return "cos(" + a->toString() + ")"; }

	std::string toJsString() const override { return "Math.cos(" + a->toJsString() + ")"; }

	bool isConstant() const override { return a->isConstant(); }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;

};
#define CosinePtr(T, a) ExpressionPtr<T>(new Cosine<T>(a))
#define CosinePtrf(a) CosinePtr(float, a)
#define CosinePtrd(a) CosinePtr(double, a)




template<typename T>
inline T Sine<T>::evaluate(T x, T y) const {
	return sin(a->evaluate(x, y));
}

template<typename T>
inline ExpressionPtr<T> Sine<T>::derivative(int dimension) const {
	// sin'(a) = a' cos(a)
	return MultiplicationPtr(T, a->derivative(dimension), CosinePtr(T, a));
}

template<typename T>
inline ExpressionPtr<T> Sine<T>::simplify() const {
	if (a->isConstant()) {
		return ConstantPtr(T, sin(a->evaluate(0, 0)));
	}
	return SinePtr(T, a->simplify());
}

template<typename T>
inline ExpressionPtr<T> Sine<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	auto newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	if (MUTATION) {
		return grammar->instantiateFunction(newA, rng);
	}
	return SinePtr(T, newA);
}

template<typename T>
inline T Cosine<T>::evaluate(T x, T y) const {
	return cos(a->evaluate(x, y));
}

template<typename T>
inline ExpressionPtr<T> Cosine<T>::derivative(int dimension) const {
	// cos'(a) = -a' sin(a)
	auto minusAPrime = MultiplicationPtr(T, ConstantPtr(T, -1), a->derivative(dimension));
	return MultiplicationPtr(T, minusAPrime, SinePtr(T, a));
}

template<typename T>
inline ExpressionPtr<T> Cosine<T>::simplify() const {
	if (a->isConstant()) {
		return ConstantPtr(T, cos(a->evaluate(0, 0)));
	}
	return CosinePtr(T, a->simplify());
}

template<typename T>
inline ExpressionPtr<T> Cosine<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	auto newA = a->mutate(rng, mutationChance, treeMutationChance, grammar, false);
	if (MUTATION) {
		return grammar->instantiateFunction(newA, rng);
	}
	return CosinePtr(T, newA);
}
