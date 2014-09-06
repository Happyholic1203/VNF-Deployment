#!/usr/bin/env python
from lib.solver import *
from lib.ds import *
from lib.utils import *

if __name__ == '__main__':
	proposedSolver = AlphaSolver(alpha=0.8)
	leastCostSolver = AlphaSolver(alpha=0.0001)
	leastVnfSolver = LeastVnfSolver()

	numFlowsPerHostSettings = [
		(1, 5),
		(5, 10),
		(10, 50),
		(100, 500),
		(1000, 5000),
		(10000, 50000)
	]

	params = [
		{
			'name': 'Small Flows: [0:0.1]',
			'numFlowsPerHostSettings': numFlowsPerHostSettings,
			'flowSettings': [{
				'minFlowAmount': 0,
				'maxFlowAmount': 0.1,
				'ratio': 1
			}]
		}, {
			'name': 'Large Flows: [0.9:1]',
			'numFlowsPerHostSettings': numFlowsPerHostSettings,
			'flowSettings': [{
				'minFlowAmount': 0.7,
				'maxFlowAmount': 1,
				'ratio': 1
			}]
		}, {
			'name': 'Medium Flows: [0.45:0.55]',
			'numFlowsPerHostSettings': numFlowsPerHostSettings,
			'flowSettings': [{
				'minFlowAmount': 0.45,
				'maxFlowAmount': 0.55,
				'ratio': 1
			}]
		}, {
			'name': 'Small and Large Flows [0:0.1] : [0.9:1] = 1 : 1',
			'numFlowsPerHostSettings': numFlowsPerHostSettings,
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
			'numFlowsPerHostSettings': numFlowsPerHostSettings,
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
			'numFlowsPerHostSettings': numFlowsPerHostSettings,
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
		for minNumFlowsPerHost, maxNumFlowsPerHost in param['numFlowsPerHostSettings']:
			results.append(TestCase(
				param['name'],
					minNumFlowsPerHost=minNumFlowsPerHost,
					maxNumFlowsPerHost=maxNumFlowsPerHost,
					flowSettings=param['flowSettings']
				).run([proposedSolver, leastCostSolver, leastVnfSolver],
					printVmPlacement=False,
					pngOutputDir='png',
					drawPng=False))

	print results
