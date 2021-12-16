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
struct Chromosome {
	std::vector<unsigned int> genes; // individual genes that make up the chromosome
	std::shared_ptr<Expression<T>> expression = nullptr; // the expression represented by the chromosome
	T fitness = INFINITY; // last fitness value computed for the chromosome
	bool parent = false; // whether the chromosome was a parent in the given generation
};

/**
 * Represents a population of genes that is being used to determine a differential equation's analytical form through GA
 */
template<typename T>
class Population {

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
	 * Probability of mutation of any single gene
	 */
	float mutationRate;

	/**
	 * Portion of the population that will be shuffled completely
	 */
	float randomMonsters;

	/**
	 * Maximum value a gene is allowed to be assigned
	 */
	int maxGeneValue;

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
	std::vector<Chromosome<T>> chromosomes;

public:

	/**
	 * Default constructor; initializes the population with random values for all of the genes
	 */
	Population(unsigned int n, unsigned int geneCount, float replicationRate, float mutationRate, float randomMonsters, const Fitness<T>* fitnessFunction, const GrammarDecoder<T>* decoder, unsigned int seed = 0, unsigned int maxGeneValue = 255);

	/**
	 * Run through a single generation of the population
	 */
	const Chromosome<T>* nextGeneration();

};




template<typename T>
inline Population<T>::Population(unsigned int n, unsigned int geneCount, float replicationRate, float mutationRate, float randomMonsters, const Fitness<T>* fitnessFunction, const GrammarDecoder<T>* decoder, unsigned int seed, unsigned int maxGeneValue) :
				replicationRate(replicationRate), mutationRate(mutationRate), randomMonsters(randomMonsters), fitnessFunction(fitnessFunction), decoder(decoder), maxGeneValue(maxGeneValue) {

	rng = std::mt19937(seed);

	assert(n >= 2);
	assert(geneCount >= 2);
	assert(replicationRate > 0 && replicationRate < 1);
	assert(mutationRate > 0 && mutationRate < 1);
	assert(fitnessFunction);
	assert(decoder);

	// Initialize population with random genes
	chromosomes = std::vector<Chromosome<T>>(n);
	for (int i = 0; i < n; ++i) {
		chromosomes[i].genes = std::vector<unsigned int>(geneCount);
		for (int j = 0; j < geneCount; ++j) {
			chromosomes[i].genes[j] = RAND % maxGeneValue;
		}
	}

}

template<typename T>
inline const Chromosome<T>* Population<T>::nextGeneration() {
	
	++generation;

	// Decode chromosomes and compute each one's fitness
	for (auto& ch : chromosomes) {
		ch.parent = false;
		ch.expression = decoder->decode(ch.genes);
		if (ch.expression == nullptr || ch.expression->isConstant()) {
			ch.fitness = INFINITY; // invalid expression, definitely don't want to keep this one
		} else {
			try {
				ch.expression = ch.expression->simplify();
				ch.fitness = fitnessFunction->fitness(ch.expression);
			} catch (...) { // handle invalid expressions with /0, log(-1), etc.
				ch.expression = nullptr;
				ch.fitness = INFINITY;
			}
		}
	}

	// Sort by fitness - best chromosomes at the top, worst at the end
	std::sort(chromosomes.begin(), chromosomes.end(), [&](const Chromosome<T>& a, const Chromosome<T>& b) -> bool {
		return a.fitness < b.fitness;
	});

	unsigned int parentCount = int(replicationRate * chromosomes.size());
	unsigned int monsterCount = int(randomMonsters * chromosomes.size());
	int crossoverCount = chromosomes.size() - monsterCount - parentCount;
	assert(crossoverCount > 0);
	assert(parentCount < chromosomes.size());
	assert(crossoverCount < chromosomes.size());
	assert(monsterCount < chromosomes.size());

	// Create crossovers
	for (size_t i = 0; i < crossoverCount / 2; ++i) { // 2 by 2, since each pair of parents creates a pair of children
		Chromosome<T>* child1 = &chromosomes[chromosomes.size() - 1 - 2 * i];
		Chromosome<T>* child2 = &chromosomes[chromosomes.size() - 2 - 2 * i];
		Chromosome<T>* parent1 = &chromosomes[0]; // one of the parents is always the best performer in the population
		Chromosome<T>* parent2 = &chromosomes[parentCount - 1]; // worst possible parent
		for (int j = 1; j < chromosomes.size() - 2 * int(crossoverCount / 2) - 2; ++j) {
			// for each chromosome between parent1 and parent2, there's a 50-50 chance that they will replace parent2
			// this mimics the exact behaviour described in the original paper, without the hassle of splitting the population into K groups
			// the parents are always the top performer and the best performer out of a random half of the full population
			if (RAND % 2 == 0) {
				parent2 = &chromosomes[j];
				break;
			}
		}
		// set up crossover; child 1 will get the first n genes from parent 1 and the last bit from parent 2, and inversely for chromosome 2
		int crossoverPosition = 1 + RAND % (parent1->genes.size() - 1);
		for (size_t j = 0; j < parent1->genes.size(); ++j) {
			child1->genes[j] = (j < crossoverPosition ? parent1 : parent2)->genes[j];
			child2->genes[j] = (j < crossoverPosition ? parent2 : parent1)->genes[j];
		}
		// mark parents as not being allowed to mutate
		parent1->parent = true;
		parent2->parent = true;
	}

	// Create "monsters", i.e. completely random chromosomes
	for (size_t i = parentCount; i < parentCount + monsterCount; ++i) {
		for (size_t j = 0; j < chromosomes[i].genes.size(); ++j) {
			chromosomes[i].genes[j] = RAND % maxGeneValue;
		}
	}

	// Create mutations
	for (auto& ch : chromosomes) {
		if (ch.parent) continue; // parents aren't allowed to mutate, which ensures we keep them in the pool for the next generation as-is
		for (size_t i = 0; i < ch.genes.size(); ++i) {
			if (RAND % 100000 > int(mutationRate * 100000)) {
				ch.genes[i] = RAND % maxGeneValue; // randomly re-assign this gene
			}
		}
	}

	// Return top performer
	return &chromosomes[0];
}

#undef RAND
