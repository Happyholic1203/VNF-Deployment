#!/usr/bin/env python
from lib.solver import *
from lib.ds import *
from lib.utils import *

if __name__ == '__main__':
	proposedSolver02 = AlphaSolver(alpha=0.2)
	proposedSolver04 = AlphaSolver(alpha=0.4)
	proposedSolver06 = AlphaSolver(alpha=0.6)
	proposedSolver08 = AlphaSolver(alpha=0.8)
	leastCostSolver = AlphaSolver(alpha=0.0001)
	leastVnfSolver = LeastVnfSolver()

	defaultNumFlowsPerHost = [
		(1, 5)
		# (5, 10),
		# (10, 50),
		# (100, 500),
		# (1000, 5000),
		# (10000, 50000)
	]

	defaultFanout = 2

	params = [
		{
			'name': 'Small Flows: [0:0.1]',
			'numFlowsPerHost': defaultNumFlowsPerHost,
			'fanout': defaultFanout,
			'flowSettings': [{
				'minFlowAmount': 0,
				'maxFlowAmount': 0.1,
				'ratio': 1
			}]
		}, {
			'name': 'Large Flows: [0.9:1]',
			'numFlowsPerHost': defaultNumFlowsPerHost,
			'fanout': defaultFanout,
			'flowSettings': [{
				'minFlowAmount': 0.7,
				'maxFlowAmount': 1,
				'ratio': 1
			}]
		}, {
			'name': 'Medium Flows: [0.45:0.55]',
			'numFlowsPerHost': defaultNumFlowsPerHost,
			'fanout': defaultFanout,
			'flowSettings': [{
				'minFlowAmount': 0.45,
				'maxFlowAmount': 0.55,
				'ratio': 1
			}]
		}, {
			'name': 'Small and Large Flows [0:0.1] : [0.9:1] = 1 : 1',
			'numFlowsPerHost': defaultNumFlowsPerHost,
			'fanout': defaultFanout,
			'flowSettings': [{
					'minFlowAmount': 0,
					'maxFlowAmount': 0.1,
					'ratio': 0.5
				}, {
					'minFlowAmount': 0.9,
					'maxFlowAmount': 1,
					'ratio': 0.5
				}
			]
		}, {
			'name': 'Small and Medium Flows [0:0.1] : [0.45:0.55] = 1 : 1',
			'numFlowsPerHost': defaultNumFlowsPerHost,
			'fanout': defaultFanout,
			'flowSettings': [{
					'minFlowAmount': 0,
					'maxFlowAmount': 0.1,
					'ratio': 0.5
				}, {
					'minFlowAmount': 0.45,
					'maxFlowAmount': 0.55,
					'ratio': 0.5
				}
			]
		}, {
			'name': 'Random Flows [0:1]',
			'numFlowsPerHost': defaultNumFlowsPerHost,
			'fanout': defaultFanout,
			'flowSettings': [{
					'minFlowAmount': 0,
					'maxFlowAmount': 1,
					'ratio': 1
				}
			]
		}
	]

	results = []

	for param in params:
		for minNumFlowsPerHost, maxNumFlowsPerHost in param['numFlowsPerHost']:
			results.append(TestCase(
					param['name'],
					minNumFlowsPerHost=minNumFlowsPerHost,
					maxNumFlowsPerHost=maxNumFlowsPerHost,
					flowSettings=param['flowSettings'],
					fanout=param['fanout']
				).run([
						leastCostSolver,
						proposedSolver02,
						proposedSolver04,
						proposedSolver06,
						proposedSolver08,
						leastVnfSolver
					],
					printVmPlacement=False,
					pngOutputDir='png',
					drawPng=False))

	print results
