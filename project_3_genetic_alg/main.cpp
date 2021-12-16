


//#define FULLY_RANDOM // whether to completely randomise the population every single generation
//#define SINGLE_EXAMPLE_ODE 3 // whether to run one example ODE problem
#define EXAMPLE_ODES // whether to run example ODE problems
#define EXAMPLE_NLODES // whether to run example NLODE problems
#define EXAMPLE_PDES // whether to run example PDE problems
#define HEAT // whether to run the heat equation problem
#define HEAT_NO_PI // whether to run the modified heat equation problem (with L=pi instead of L=1, eliminating pi from the solution)
//#define VERBOSE // whether to output console messages while training each time a new best fit is found amongst the population
#define JSON // whether to output a json file for each executed run
#define TREE_CHROMOSOMES // whether to use a TreePopulation instead of the grammar-based population
#define MULTI_RUN // whether to run each problem 50 times instead of once, with a random seed each time


#ifdef FULLY_RANDOM
	#undef TREE_CHROMOSOMES
	#define POPULATION_SIZE 5000
	#define CHROMOSOME_SIZE 50
	#define REPLICATION_RATE 0.0002
	#define MUTATION_RATE 0.0002
	#define RANDOM_RATE 0.9996
	#define GENERATIONS 1000
#else
	#define POPULATION_SIZE 5000
	#define REPLICATION_RATE 0.05
	#define MUTATION_RATE 0.1
	#define GENERATIONS 5000
#ifdef TREE_CHROMOSOMES
	#define REPLICATION_BIAS 25
	#define TREE_MUTATION_RATE 0.1
	#define RANDOM_RATE 0.1
#else
	#define CHROMOSOME_SIZE 50
	#define RANDOM_RATE 0.0
#endif
#endif


// ------------------------------------


#include <ctime>
#include <stdlib.h>
#include <cmath>
#include <thread>
#ifndef M_PI
	#define M_PI 3.14159265359
#endif
#include "Vars.h"
#include "Addition.h"
#include "Multiplication.h"
#include "Division.h"
#include "Subtraction.h"
#include "Trig.h"
#include "Exponential.h"
#include "Logarithm.h"
#include "SquareRoot.h"
#include "Power.h"
#include "GrammarDecoder.h"
#include "Population.h"
#include "TreePopulation.h"
#if defined(EXAMPLE_ODES) or defined(EXAMPLE_NLODES) or defined(SINGLE_EXAMPLE_ODE)
	#include "ExampleODEs.h"
#endif
#ifdef EXAMPLE_PDES
	#include "ExamplePDEs.h"
#endif
#include "FileWriter.h"



Fitness<double> heatPde(double tMax) {
	// exact solution:
	// u(x, t) = e^{- pi^2 t} sin(pi x)
	return Fitness<double>(
		[](FunctionParams<double> p) -> const double {
			return p.ddx2 - p.ddy; // d^2/dx^2 u = d/dt u
		},
		Domain<double>(0, 1, 50), Domain<double>(0, tMax, 50), 100,
		{
			Boundary<double>(0, 0, [](double t, double f, double dfdx, double ddfdx) -> double {
				return -f; // u(0, t) = 0
			}),
			Boundary<double>(1, 0, [](double t, double f, double dfdx, double ddfdx) -> double {
				return -f; // u(L, t) = 0
			}),
			Boundary<double>(0, 1, [](double x, double f, double dfdt, double ddfdt) -> double {
				return f - sin(M_PI * x); // u(x, 0) = sin(pi x)
			})
		}
	);
}

Fitness<double> heatPdeNoPi(double tMax) {
	// exact solution:
	// u(x, t) = e^{-t} sin(x)
	return Fitness<double>(
		[](FunctionParams<double> p) -> const double {
			return p.ddx2 - p.ddy;
		},
		Domain<double>(0, M_PI, 50), Domain<double>(0, tMax, 50), 100,
		{
			Boundary<double>(0, 0, [](double t, double f, double dfdx, double ddfdx) -> double {
				return -f; // u(0, t) = 0
			}),
			Boundary<double>(M_PI, 0, [](double t, double f, double dfdx, double ddfdx) -> double {
				return -f; // u(pi, t) = 0
			}),
			Boundary<double>(0, 1, [](double x, double f, double dfdt, double ddfdt) -> double {
				return f - sin(x); // u(x, 0) = sin(x)
			})
		}
	);
}




/**
 * Solves an ODE/PDE given its fitness function and a grammatical decoder
 * This uses the parameters #define'd at the top of main.cpp
 */
void solve(std::string name, Fitness<double> fitnessFunction, GrammarDecoder<double>* decoder, int seed) {

#ifdef MULTI_RUN
	int ogSeed = seed;
	for (; seed < ogSeed + 50; ++seed) {
#endif

		// Init population
#ifdef TREE_CHROMOSOMES
		TreePopulation<double> population(POPULATION_SIZE, REPLICATION_RATE, REPLICATION_BIAS, MUTATION_RATE, TREE_MUTATION_RATE, RANDOM_RATE, &fitnessFunction, decoder, seed);
		const TreeChromosome<double>* top = nullptr;
#else
		Population<double> population(POPULATION_SIZE, CHROMOSOME_SIZE, REPLICATION_RATE, MUTATION_RATE, RANDOM_RATE, &fitnessFunction, decoder, seed);
		const Chromosome<double>* top = nullptr;
#endif


		// create json string with results (hard-coded json structure since it's kept fairly simple)
#ifdef JSON
		std::string json = "{\"time\":" + std::to_string(time(nullptr)) + ",\"seed\":" + std::to_string(seed) + ",\"problem\":\"" + name + "\",\"useTrees\":";
#ifdef TREE_CHROMOSOMES
		json += "true,";
#else
		json += "false,";
#endif
		json += "\"populationSize\":" + std::to_string(POPULATION_SIZE) + ",";
		json += "\"replicationRate\":" + std::to_string(REPLICATION_RATE) + ",";
		json += "\"mutationRate\":" + std::to_string(MUTATION_RATE) + ",";
		json += "\"maxGeneration\":" + std::to_string(GENERATIONS) + ",";
#ifdef TREE_CHROMOSOMES
		json += "\"replicationBias\":" + std::to_string(REPLICATION_BIAS) + ",";
		json += "\"treeMutationRate\":" + std::to_string(TREE_MUTATION_RATE) + ",";
#else
		json += "\"chromosomeSize\":" + std::to_string(CHROMOSOME_SIZE) + ",";
		json += "\"randomRate\":" + std::to_string(RANDOM_RATE) + ",";
#endif
		json += "\"generations\":[";
#endif


		// Iterate over generations
		int gen;
		double fitness = INFINITY;
		std::shared_ptr<Expression<double>> bestExpression = nullptr;
		for (gen = 1; gen <= GENERATIONS; ++gen) {
			top = population.nextGeneration();
			if (top && top->fitness < fitness) {
				fitness = top->fitness;
				bestExpression = top->expression;
#ifdef VERBOSE
				printf("%s \tGen. %d, \tfitness %f, \tf(x, y) = %s\n", name.c_str(), gen, fitness, top->expression->toString().c_str());
#endif
#ifdef JSON // add one json object to the array of generations each time a new best fit is found
				json += "{\"generation\":" + std::to_string(gen) + ",\"fitness\":" + std::to_string(top->fitness) + ",";
				json += "\"expression\":\"" + top->expression->toString() + "\",\"jsExpression\":\"" + top->expression->toJsString() + "\"},";
#endif
			}
			if (top && top->fitness < 1e-7) {
				break;
			}
		}


		// Log result
		if (!top) {
			printf("Could not solve %s, null result.\n\n", name.c_str());
		} else {
			printf("\nFinished solving %s in %d generations: \tfitness %f, \tf(x, y) = %s\n\n", name.c_str(), gen, fitness, bestExpression->toString().c_str());
			printf("d/dx f(x, y) = %s\n", bestExpression->derivative(0)->simplify()->toString().c_str());
			printf("d/dy f(x, y) = %s\n", bestExpression->derivative(1)->simplify()->toString().c_str());
			printf("d^2/dx^2 f(x, y) = %s\n", bestExpression->derivative(0)->derivative(0)->simplify()->toString().c_str());
			printf("d^2/dy^2 f(x, y) = %s\n\n", bestExpression->derivative(1)->derivative(1)->simplify()->toString().c_str());
		}


		// close json string and output to file
#ifdef JSON
		json = json.substr(0, json.length() - 1); // remove last trailing comma
		json += "]}";
		FileWriter::Write("results/" + name + "_" + std::to_string(time(nullptr)) + ".json", json);
#endif

#ifdef MULTI_RUN
	}
#endif
}




int main() {

	// Set up grammar - two different variants for 1D problems (ODEs) and 2D problems (PDEs)
	std::vector<GrammaticalElement_base<double>*> variables1d = {
		Gd(VarX)
	};
	std::vector<GrammaticalElement_base<double>*> variables2d = {
		Gd(VarX),
		Gd(VarY)
	};
	std::vector<GrammaticalElement_base<double>*> operations = {
		G2d(Addition),
		G2d(Subtraction),
		G2d(Multiplication),
		G2d(Division),
		G2d(Power)
	};
	std::vector<GrammaticalElement_base<double>*> functions = {
		G1d(Sine),
		G1d(Cosine),
		G1d(Exponential),
		G1d(Logarithm),
		// G1d(SquareRoot) <- not including sqrt() in the grammar since it's not quite general enough, we already have powers so it's possible to obtain ^0.5 anyways
	};
	std::vector<double> constants = {
		-1, 0, 1, 2, 3, M_PI, 4, 5, 6, 7, 8, 9, 10
	};
	auto decoder1d = new GrammarDecoder<double>(0, variables1d, operations, functions, constants);
	auto decoder2d = new GrammarDecoder<double>(0, variables2d, operations, functions, constants);


	std::vector<std::thread*> threads;



	// solve example ODEs from the original paper
#if defined(SINGLE_EXAMPLE_ODE) and not defined(EXAMPLE_ODES)
	threads.push_back(new std::thread(solve, "ODE" + std::to_string(SINGLE_EXAMPLE_ODE), getExampleODE(SINGLE_EXAMPLE_ODE), decoder1d, 0));
#endif
#ifdef EXAMPLE_ODES
	for (int i = 1; i <= 9; ++i) {
		threads.push_back(new std::thread(solve, "ODE" + std::to_string(i), getExampleODE(i), decoder1d, i));
	}
#endif

	// solve example NLODEs from the original paper
#ifdef EXAMPLE_NLODES
	for (int i = 1; i <= 4; ++i) {
		threads.push_back(new std::thread(solve, "NLODE" + std::to_string(i), getExampleNLODE(i), decoder1d, i));
	}
#endif

	// solve example PDEs from the original paper
#ifdef EXAMPLE_PDES
	for (int i = 1; i <= 6; ++i) {
		threads.push_back(new std::thread(solve, "PDE" + std::to_string(i), getExamplePDE(i), decoder2d, i));
	}
#endif

	// solve 1D temporal heat equation problem
#ifdef HEAT
	threads.push_back(new std::thread(solve, "Heat", heatPde(1), decoder2d, 1337));
#endif
#ifdef HEAT_NO_PI
	threads.push_back(new std::thread(solve, "Heat[-pi]", heatPdeNoPi(1), decoder2d, 1337));
#endif


	// Run all threads until they terminate, then exit the program
	for (auto it = threads.begin(); it != threads.end(); it++) {
		(*it)->join();
		delete* it;
	}

	delete decoder1d;
	delete decoder2d;
	return 0;
}
