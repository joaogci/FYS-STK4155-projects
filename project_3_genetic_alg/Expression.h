#pragma once

#include <array>
#include <string>
#include <memory>
#include <random>
#include <cmath>

#define TREE_MUTATION() if (abs(int(rng())) % 10000 < int(treeMutationChance * 10000)) return grammar->instantiateExpression(rng);
#define MUTATION (abs(int(rng())) % 10000 < int(mutationChance * 10000))
#define RAND_NEG1_1 double(abs(int(rng)) % 1000) / 500 - 0.5

template<typename T>
class GrammarDecoder;

template<typename T>
class Expression {
public:

	/**
	 * Evaluates the expression at point x
	 */
	virtual T evaluate(T x, T y) const = 0;

	/**
	 * Returns the expression that corresponds to the first derivative of this expression with respect to some dimensional variable (x if dimension == 0, y if dimension == 1, etc)
	 */
	virtual const std::shared_ptr<Expression<T>> derivative(int dimension) const = 0;

	/**
	 * Simplifies the given expression to make it easier to write out
	 * This will most likely NOT produce the best possible simplification of the given expression, but will at least be an improvement
	 */
	virtual const std::shared_ptr<Expression<T>> simplify() const = 0;

	/**
	 * Returns a string representation of the expression
	 */
	virtual std::string toString() const = 0;

	/**
	 * Returns a string representation of the expression that can be evaluated by JavaScript without further modifications
	 */
	virtual std::string toJsString() const = 0;

	/**
	 * Returns whether the given expression is constant (i.e. y(x) = y(0) for all x in R)
	 */
	virtual bool isConstant() const = 0;

	/**
	 * Potentially mutates the expression or one of its sub-nodes with a random probability, for use in genetic algorithms
	 */
	virtual const std::shared_ptr<Expression<T>> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const = 0;

};
template<typename T>
using ExpressionPtr = const std::shared_ptr<Expression<T>>;




// Define constants here as well since they'll be needed in most dependents

template<typename T>
class Constant : public Expression<T> {
private:
	T v;

public:

	Constant(T v) : v(v) {}
	Constant(int v) : v(T(v)) {}

	T evaluate(T x, T y) const override;

	ExpressionPtr<T> derivative(int dimension) const override;

	ExpressionPtr<T> simplify() const override;

	std::string toString() const override { return std::to_string(v); }

	std::string toJsString() const override { return std::to_string(v); }

	bool isConstant() const override { return true; }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;
};
#define ConstantPtr(T, v) ExpressionPtr<T>(new Constant<T>(v))
#define ConstantPtrf(v) ConstantPtr(float, v)
#define ConstantPtrd(v) ConstantPtr(double, v)




template<typename T>
inline T Constant<T>::evaluate(T x, T y) const {
	return v;
}

template<typename T>
inline ExpressionPtr<T> Constant<T>::derivative(int dimension) const {
	return ExpressionPtr<T>(new Constant<T>(0));
}

template<typename T>
inline ExpressionPtr<T> Constant<T>::simplify() const {
	return ExpressionPtr<T>(new Constant<T>(v));
}

template<typename T>
inline ExpressionPtr<T> Constant<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	if (MUTATION) {
		// either modify the value by a little bit (normal distrib), or pick a new constant altogether
		if (abs(int(rng())) % 2 == 0) {
			std::normal_distribution<T> n(0, 1);
			return ConstantPtr(T, v + n(rng));
		} else {
			return grammar->instantiateConstant(rng);
		}
	}
	return ConstantPtr(T, v);
}

