#pragma once

#pragma once

#include <cassert>
#include <vector>
#include <algorithm>
#include <random>
#include "GrammarDecoder.h"
#include "Fitness.h"
#include "Expression.h"

#define RAND abs(int(rng()))

/**
 * Represents a single chromosome (i.e. individual) in the population
 */
template<typename T>
struct TreeChromosome {
	std::shared_ptr<Expression<T>> expression = nullptr; // the expression represented by the chromosome
	T fitness = INFINITY; // last fitness value computed for the chromosome
};

/**
 * Represents a population of genes that is being used to determine a differential equation's analytical form through GA
 */
template<typename T>
class TreePopulation {

private:

	/**
	 * Random number generator to use for the population
	 */
	std::mt19937 rng;

	/**
	 * Gives the number of generations that the population has gone through
	 */
	int generation = 0;

	/**
	 * Percentage of individuals that will create children
	 */
	float replicationRate;

	/**
	 * Each consecutive parent chromosome, ordered from best to worst, has a 1/replicationBias chance to replicate into any one single child
	 * i.e. if replicationBias = 2, the top performer has 1/2 chance, the second has 1/4, third has 1/8, etc
	 *		if replicationBias = 5, the top performer has 1/5 chance (0.2), the second has 4/25 (0.16), third has 16/125 (0.128), etc
	 *		if replicationBias = 20, the top performer has 1/20 chance (0.05), the second has 19/400 (0.0475), third has 361/8000 (0.045125), etc
	 */
	int replicationBias;

	/**
	 * Probability of mutation of any single gene
	 */
	float mutationRate;

	/**
	 * Probability of mutation of a sub-tree
	 */
	float treeMutationRate;

	/**
	 * Proportion of the population that will be replaced with completely random expressions each generation
	 */
	float randomRate;

	/**
	 * Fitness function, which encodes the problem at hand
	 */
	const Fitness<T>* fitnessFunction;

	/**
	 * Decoder to use to read mathematical expressions from chromosomes
	 */
	const GrammarDecoder<T>* decoder;

	/**
	 * The actual population, made up of a collection of chromosomes
	 */
	std::vector<TreeChromosome<T>> chromosomes;

public:

	/**
	 * Default constructor; initializes the population with random values for all of the genes
	 */
	TreePopulation(unsigned int n, float replicationRate, int replicationBias, float mutationRate, float treeMutationRate, float randomRate, const Fitness<T>* fitnessFunction, const GrammarDecoder<T>* decoder, unsigned int seed = 0);

	/**
	 * Run through a single generation of the population
	 */
	const TreeChromosome<T>* nextGeneration();

};




template<typename T>
inline TreePopulation<T>::TreePopulation(unsigned int n, float replicationRate, int replicationBias, float mutationRate, float treeMutationRate, float randomRate, const Fitness<T>* fitnessFunction, const GrammarDecoder<T>* decoder, unsigned int seed) :
			replicationRate(replicationRate), replicationBias(replicationBias), mutationRate(mutationRate), treeMutationRate(treeMutationRate), randomRate(randomRate), fitnessFunction(fitnessFunction), decoder(decoder) {

	rng = std::mt19937(seed);

	assert(n >= 2);
	assert(fitnessFunction);
	assert(decoder);

	// Initialize population with random genes using the grammar
	chromosomes = std::vector<TreeChromosome<T>>(n);
	for (int i = 0; i < n; ++i) {
		// force initial expressions of the form 1 * (...)
		chromosomes[i].expression = MultiplicationPtr(T, ConstantPtr(T, 1), decoder->instantiateExpression(rng, 5));
	}

}

template<typename T>
inline const TreeChromosome<T>* TreePopulation<T>::nextGeneration() {

	++generation;

	// Compute each chromosome's fitness
	for (auto& ch : chromosomes) {
		if (ch.expression == nullptr || ch.expression->isConstant()) {
			ch.fitness = INFINITY; // invalid expression, definitely don't want to keep this one
		} else {
			try {
				//ch.expression = ch.expression->simplify();
				ch.fitness = fitnessFunction->fitness(ch.expression);
			} catch (...) { // handle invalid expressions with /0, log(-1), etc.
				ch.fitness = INFINITY;
			}
		}
	}

	// Sort by fitness - best chromosomes at the top, worst at the end
	std::sort(chromosomes.begin(), chromosomes.end(), [&](const TreeChromosome<T>& a, const TreeChromosome<T>& b) -> bool {
		return a.fitness < b.fitness;
	});

	// Genetic operations

	// Replication
	unsigned int parentCount = int(replicationRate * chromosomes.size());
	unsigned int randomCount = int(randomRate * chromosomes.size());
	for (int i = parentCount; i < chromosomes.size() - randomCount; ++i) {
		// replace chromosome with a parent selected at random
		for (int j = 0; j < parentCount; ++j) {
			if (RAND % replicationBias == 0 || j == parentCount-1) {

				// replicate chromosome [j]
				chromosomes[i].expression = chromosomes[j].expression;

				// mutations - note that we only mutate children, not parents

				// modify random nodes and subtrees in expression
				chromosomes[i].expression = chromosomes[i].expression->mutate(rng, mutationRate, treeMutationRate, decoder, true);

				break;
			}
		}
	}

	// Random individuals
	for (int i = chromosomes.size() - randomCount; i < chromosomes.size(); ++i) {
		chromosomes[i].expression = MultiplicationPtr(T, ConstantPtr(T, 1), decoder->instantiateExpression(rng, 5));
	}

	// Return top performer
	return &chromosomes[0];
}


#undef RAND
