from solver import *
from ds import *
import random
import time

class TestCase(object):
    def __init__(self, name, flowSettings=None,
        depth=3, fanout=2, minFlowAmount=0, maxFlowAmount=1,
        minNumFlowsPerHost=2, maxNumFlowsPerHost=5):
        if not self.checkFlowSettings(flowSettings):
            raise Exception('Invalid flow settings: %r' % (flowSettings))

        self.name = name
        self.flowSettings = flowSettings
        self.minNumFlowsPerHost = minNumFlowsPerHost
        self.maxNumFlowsPerHost = maxNumFlowsPerHost
        self.depth = depth
        self.fanout = fanout

        if self.flowSettings is None:
            self.flowSettings = [
                {
                    'minFlowAmount': minFlowAmount,
                    'maxFlowAmount': maxFlowAmount,
                    'ratio': 1
                }
            ]

        self.tree = self.buildTree(self.depth, self.fanout)

    def reset(self):
        self.tree.reset()
        return self

    def buildTree(self, depth, fanout):
        tree = Tree(depth=depth, fanout=fanout)
        hosts = tree.getHosts()
        for h in hosts:
            for _ in range(self.getNumFlowsPerHost()):
                h.addFlow(self.getFlowAmount())
        return tree

    def getNumFlowsPerHost(self):
        return random.randint(
            self.minNumFlowsPerHost,
            self.maxNumFlowsPerHost)

    def getFlowAmount(self):
        uniform = random.uniform(0, 1.0)
        accumulatedProb = 0
        for flowSetting in self.flowSettings:
            accumulatedProb += flowSetting['ratio']
            if accumulatedProb >= uniform:
                return random.uniform(flowSetting['minFlowAmount'], flowSetting['maxFlowAmount'])
        raise Exception('The ratio in flowSettings do not add up to 1.0')

    # TODO
    def run(self, solvers, drawPng=None, pngOutputDir=None, printVmPlacement=True, printSummary=True):
        for s in solvers:
            assert isinstance(s, Solver)
        if not pngOutputDir:
            pngOutputDir = '.'

        allFlows = self.tree.getFlows()
        numFlows = len(allFlows)
        numHosts = len(self.tree.getHosts())
        flowAmounts = [f.getCapacity() for f in allFlows]
        flowsPerHostList = [len(flows) for flows in [h.getFlows() for h in self.tree.getHosts()]]
        totalFlowAmount = self.tree.getTotalFlowAmount()
        print '** TestCase: %s' % self.name
        print '** Number of Hosts: %d' % numHosts
        print '** Number of Flows per Host: (min, max, avg) = (%d, %d, %.2f)' % (
            min(flowsPerHostList),
            max(flowsPerHostList),
            (float(numFlows) / float(numHosts)))
        print '** Total Number of Flows: %d' % numFlows
        print '** Flow Amount: (min, max, avg) = (%.2f, %.2f, %.2f)' % (
            min(flowAmounts),
            max(flowAmounts),
            (totalFlowAmount / float(numFlows)))
        print '** Total Flow Amount: %.2f' % totalFlowAmount

        result = {
            'name': self.name,
            'numHosts': numHosts,
            'flowsPerHost': {
                'min': min(flowsPerHostList),
                'max': max(flowsPerHostList),
                'avg': (float(numFlows) / float(numHosts))
            },
            'totalNumFlows': numFlows,
            'flowAmount': {
                'min': min(flowAmounts),
                'max': max(flowAmounts),
                'avg': (totalFlowAmount / float(numFlows))
            },
            'totalFlowAmount': totalFlowAmount
        }

        records = []
        for solver in solvers:
            startTime = time.time()
            solver.solve(self.tree)
            print '** Runtime [%s]: %.2f seconds' % \
                (solver.getName(), time.time() - startTime)
            if drawPng:
                self.tree.draw(
                    '%s/%s(%s).png' % (pngOutputDir, self.name, solver.getName()),
                    showFlowCapacity=True,
                    showFlowVm=True)
            records.append({
                'solver': solver.getName(),
                'numVms': solver.getNumVms(),
                'cost': solver.getTotalCost()
            })
            # print solver.getSolution(
            #   showVmPlacement=printVmPlacement,
            #   showSummary=printSummary)
            self.reset()
            solver.reset()

        result['records'] = records
        self.printRecords(records)
        return result

    @staticmethod
    def printRecords(records):
        fmt = '{:>20}{:>10}{:>10}'
        print fmt.format('solver', 'numVms', 'cost')
        print '---------------------------------------------'
        for rec in records:
            print fmt.format(
                rec['solver'],
                rec['numVms'],
                '%.2f' % rec['cost'])
        print ''

    # TODO
    def checkFlowSettings(self, flowSettings):
        if not flowSettings:
            return True
        return True