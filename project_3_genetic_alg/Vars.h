#pragma once

#include "Expression.h"



template<typename T>
class VarX : public Expression<T> {
public:

	inline T evaluate(T x, T y) const override { return x; }

	inline ExpressionPtr<T> derivative(int dimension) const override { return ExpressionPtr<T>(new Constant<T>(dimension == 0)); }

	inline ExpressionPtr<T> simplify() const override { return ExpressionPtr<T>(new VarX<T>()); }

	std::string toString() const override { return "x"; }

	std::string toJsString() const override { return toString(); }

	bool isConstant() const override { return false; }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;

};
#define VarXPtr(T) ExpressionPtr<T>(new VarX<T>())
#define VarXPtrf VarXPtr(float)
#define VarXPtrd VarXPtr(double)



template<typename T>
class VarY : public Expression<T> {
public:

	inline T evaluate(T x, T y) const override { return y; }

	inline ExpressionPtr<T> derivative(int dimension) const override { return ExpressionPtr<T>(new Constant<T>(dimension == 1)); }

	inline ExpressionPtr<T> simplify() const override { return ExpressionPtr<T>(new VarY<T>()); }

	std::string toString() const override { return "y"; }

	std::string toJsString() const override { return toString(); }

	bool isConstant() const override { return false; }

	ExpressionPtr<T> mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const override;

};
#define VarYPtr(T) ExpressionPtr<T>(new VarY<T>())
#define VarYPtrf VarYPtr(float)
#define VarYPtrd VarYPtr(double)

template<typename T>
inline ExpressionPtr<T> VarX<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	if (MUTATION) {
		return grammar->instantiateVar(rng);
	}
	return VarXPtr(T);
}

template<typename T>
inline ExpressionPtr<T> VarY<T>::mutate(std::mt19937& rng, double mutationChance, double treeMutationChance, const GrammarDecoder<T>* grammar, bool first) const {
	TREE_MUTATION();
	if (MUTATION) {
		return grammar->instantiateVar(rng);
	}
	return VarYPtr(T);
}
