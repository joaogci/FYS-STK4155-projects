#pragma once

#include <vector>
#include <cassert>
#include "Expression.h"


// Helper macros to set up a grammar
#define G2f(t) new GrammaticalElement2Args<t<float>, float>()
#define G1f(t) new GrammaticalElement1Arg<t<float>, float>()
#define Gf(t) new GrammaticalElement0Args<t<float>, float>()
#define G2d(t) new GrammaticalElement2Args<t<double>, double>()
#define G1d(t) new GrammaticalElement1Arg<t<double>, double>()
#define Gd(t) new GrammaticalElement0Args<t<double>, double>()


// Token class to denote an individual grammatical element, i.e. an operation that will take n parameters
template<typename T>
class GrammaticalElement_base {
public:
	virtual const ExpressionPtr<T> instantiate0Args() = 0;
	virtual const ExpressionPtr<T> instantiate1Arg(const std::shared_ptr<Expression<T>> a) = 0;
	virtual const ExpressionPtr<T> instantiate2Args(const std::shared_ptr<Expression<T>> a, const std::shared_ptr<Expression<T>> b) = 0;
};

template<typename ChildExpression, typename T>
class GrammaticalElement0Args : public GrammaticalElement_base<T> {
public:
	/**
	 * Creates an instance of the templated expression type; either the 0-, 1- or 2-arguments version should be called, depending on what the expression type expects
	 */
	const ExpressionPtr<T> instantiate0Args() override {
		return std::shared_ptr<ChildExpression>(new ChildExpression());
	}
	const ExpressionPtr<T> instantiate1Arg(const std::shared_ptr<Expression<T>> a) override {
		return nullptr;
	}
	const ExpressionPtr<T> instantiate2Args(const std::shared_ptr<Expression<T>> a, const std::shared_ptr<Expression<T>> b) override {
		return nullptr;
	}
}; // class GrammaticalElement0Args
template<typename ChildExpression, typename T>
class GrammaticalElement1Arg : public GrammaticalElement_base<T> {
public:
	/**
	 * Creates an instance of the templated expression type; either the 0-, 1- or 2-arguments version should be called, depending on what the expression type expects
	 */
	const ExpressionPtr<T> instantiate0Args() override {
		return nullptr;
	}
	const ExpressionPtr<T> instantiate1Arg(const std::shared_ptr<Expression<T>> a) override {
		return std::shared_ptr<ChildExpression>(new ChildExpression(a));
	}
	const ExpressionPtr<T> instantiate2Args(const std::shared_ptr<Expression<T>> a, const std::shared_ptr<Expression<T>> b) override {
		return nullptr;
	}
}; // class GrammaticalElement1Arg
template<typename ChildExpression, typename T>
class GrammaticalElement2Args : public GrammaticalElement_base<T> {
public:
	/**
	 * Creates an instance of the templated expression type; either the 0-, 1- or 2-arguments version should be called, depending on what the expression type expects
	 */
	const ExpressionPtr<T> instantiate0Args() override {
		return nullptr;
	}
	const ExpressionPtr<T> instantiate1Arg(const std::shared_ptr<Expression<T>> a) override {
		return nullptr;
	}
	const ExpressionPtr<T> instantiate2Args(const std::shared_ptr<Expression<T>> a, const std::shared_ptr<Expression<T>> b) override {
		return std::shared_ptr<ChildExpression>(new ChildExpression(a, b));
	}
}; // class GrammaticalElement2Args


/**
 * Can be used to decode sequences of integers into valid mathematical expressions
 */
template<typename T>
class GrammarDecoder {
private:

	/**
	 * The maximum number of times the pointer will be allowed to wrap around the given sequence before declaring it invalid
	 * For example, a sequence of size 3 might produce an output that expects 5 elements, in which case if maxWraparounds is 1 or more, the sequence will be found valid
	 */
	const int maxWraparounds;

	/**
	 * The library of base operations and functions to allow in expressions, respectively
	 */
	const std::vector<GrammaticalElement_base<T>*> variables;
	const std::vector<GrammaticalElement_base<T>*> operations;
	const std::vector<GrammaticalElement_base<T>*> functions;
	const std::vector<T> constants;

public:

	/**
	 * Default constructor
	 */
	GrammarDecoder(
		const int& maxWraparounds,
		const std::vector<GrammaticalElement_base<T>*>& variables,
		const std::vector<GrammaticalElement_base<T>*>& operations,
		const std::vector<GrammaticalElement_base<T>*>& functions,
		const std::vector<T>& constants
	) : maxWraparounds(maxWraparounds), variables(variables), operations(operations), functions(functions), constants(constants) {}

	/**
	 * Decodes a sequence of integers into a valid mathematical expression
	 */
	const ExpressionPtr<T> decode(const std::vector<unsigned int>& sequence) const;

	/**
	 * Instantiates and returns a random function/operation/var on a random probability
	 */
	const ExpressionPtr<T> instantiateFunction(const ExpressionPtr<T> a, std::mt19937& rng) const;
	const ExpressionPtr<T> instantiateOperation(const ExpressionPtr<T> a, const ExpressionPtr<T> b, std::mt19937& rng) const;
	const ExpressionPtr<T> instantiateVar(std::mt19937& rng) const;
	const ExpressionPtr<T> instantiateConstant(std::mt19937& rng) const;
	const ExpressionPtr<T> instantiateExpression(std::mt19937& rng, int maxDepth = 3, int depth = 0) const;

private:

	/**
	 * Decodes a single expression recursively; returns false if the sequence is found invalid
	 */
	bool decodeExpression(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outExpression) const;

	/**
	 * Decodes an operation recursively; returns false if the sequence is found invalid
	 */
	bool decodeOperation(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outOperation) const;

	/**
	 * Decodes a function recursively; returns false if the sequence is found invalid
	 */
	bool decodeFunction(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outFunction) const;

	/**
	 * Decodes a dimensional variable (x, y, z, etc)
	 */
	bool decodeVariable(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outVariable) const;

	/**
	 * Decodes a constant recursively; returns false if the sequence is found invalid
	 */
	bool decodeConstant(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outConstant) const;

};// class GrammarDecoder



template<typename T>
inline const ExpressionPtr<T> GrammarDecoder<T>::decode(const std::vector<unsigned int>& sequence) const {

	unsigned int ptr = 0;
	unsigned int wraps = 0;
	std::shared_ptr<Expression<T>> expression;
	
	// Attempt to decode the whole sequence as an overarching expression
	if (!decodeExpression(sequence, ptr, wraps, expression)) return nullptr;

	return expression;
}

template<typename T>
inline const ExpressionPtr<T> GrammarDecoder<T>::instantiateFunction(const ExpressionPtr<T> a, std::mt19937& rng) const {
	return functions[abs(int(rng())) % functions.size()]->instantiate1Arg(a);
}

template<typename T>
inline const ExpressionPtr<T> GrammarDecoder<T>::instantiateOperation(const ExpressionPtr<T> a, const ExpressionPtr<T> b, std::mt19937& rng) const {
	return operations[abs(int(rng())) % operations.size()]->instantiate2Args(a, b);
}

template<typename T>
inline const ExpressionPtr<T> GrammarDecoder<T>::instantiateVar(std::mt19937& rng) const {
	return variables[abs(int(rng())) % variables.size()]->instantiate0Args();
}

template<typename T>
inline const ExpressionPtr<T> GrammarDecoder<T>::instantiateConstant(std::mt19937& rng) const {
	return ConstantPtr(T, constants[abs(int(rng())) % constants.size()]);
}

template<typename T>
inline const ExpressionPtr<T> GrammarDecoder<T>::instantiateExpression(std::mt19937& rng, int maxDepth, int depth) const {
	if (depth >= maxDepth) { // prevent creating sub-trees too deeply nested
		return abs(int(rng())) % 2 == 0 ? instantiateConstant(rng) : instantiateVar(rng);
	}
	switch (abs(int(rng())) % 4) {
	case 0:
		return instantiateConstant(rng);
	case 1:
		return instantiateVar(rng);
	case 2:
		return instantiateFunction(instantiateExpression(rng, maxDepth, depth+1), rng);
	default:
		return instantiateOperation(instantiateExpression(rng, maxDepth, depth+1), instantiateExpression(rng, maxDepth, depth+1), rng);
	}
}

template<typename T>
inline bool GrammarDecoder<T>::decodeExpression(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outExpression) const {

	// walk through sequence
	int head = sequence[ptr];
	++ptr;
	if (ptr >= sequence.size()) {
		ptr = 0;
		++wraps;
		if (wraps >= maxWraparounds) {
			return false;
		}
	}

	switch (head % 4) {
	case 0:
		// operation
		if(!decodeOperation(sequence, ptr, wraps, outExpression)) return false;
		break;
	case 1:
		// function
		if(!decodeFunction(sequence, ptr, wraps, outExpression)) return false;
		break;
	case 2:
		// digit
		if(!decodeConstant(sequence, ptr, wraps, outExpression)) return false;
		break;
	case 3:
		// x, y, z, ... (depending on dimensionality of problem)
		if (!decodeVariable(sequence, ptr, wraps, outExpression)) return false;
		break;
	}

	return true;
}

template<typename T>
inline bool GrammarDecoder<T>::decodeOperation(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outOperation) const {

	// obtain the two expressions to place on either side of the operation
	std::shared_ptr<Expression<T>> a, b;
	if (!decodeExpression(sequence, ptr, wraps, a)) return false;
	
	// walk through sequence - for operations, the first operand is given first (before the actual operation type)
	int head = sequence[ptr];
	++ptr;
	if (ptr >= sequence.size()) {
		ptr = 0;
		++wraps;
		if (wraps >= maxWraparounds) {
			return false;
		}
	}

	if (!decodeExpression(sequence, ptr, wraps, b)) return false;

	// decode the head into one of the available operations
	outOperation = operations[head % operations.size()]->instantiate2Args(a, b);
	assert(outOperation);

	return true;
}

template<typename T>
inline bool GrammarDecoder<T>::decodeFunction(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outFunction) const {

	// walk through sequence
	int head = sequence[ptr];
	++ptr;
	if (ptr >= sequence.size()) {
		ptr = 0;
		++wraps;
		if (wraps >= maxWraparounds) {
			return false;
		}
	}

	// obtain the expression that lives inside the function
	std::shared_ptr<Expression<T>> inner;
	if (!decodeExpression(sequence, ptr, wraps, inner)) return false;

	// decode the head into one of the available functions
	outFunction = functions[head % functions.size()]->instantiate1Arg(inner);
	assert(outFunction);

	return true;
}

template<typename T>
inline bool GrammarDecoder<T>::decodeVariable(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outVariable) const {

	// walk through sequence
	int head = sequence[ptr];
	++ptr;
	if (ptr >= sequence.size()) {
		ptr = 0;
		++wraps;
		if (wraps >= maxWraparounds) {
			return false;
		}
	}

	// decode the head into one of the available functions
	outVariable = variables[head % variables.size()]->instantiate0Args();
	assert(outVariable);

	return true;
}

template<typename T>
inline bool GrammarDecoder<T>::decodeConstant(const std::vector<unsigned int>& sequence, unsigned int& ptr, unsigned int& wraps, std::shared_ptr<Expression<T>>& outConstant) const {

	// walk through sequence
	int head = sequence[ptr];
	++ptr;
	if (ptr >= sequence.size()) {
		ptr = 0;
		++wraps;
		if (wraps >= maxWraparounds) {
			return false;
		}
	}

	// one of the available constants
	outConstant = ConstantPtr(T, constants[head % constants.size()]);

	return true;
}
