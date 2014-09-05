#!/usr/bin/env python

import pydot
import Queue
import random

class TreeNode(object):
	__id = 0
	dotSettings = {
		'style': 'filled',
		'fillcolor': 'white'
	}
	def __init__(self, **attrs):
		self.attrs = attrs
		self.parent = None
		self.children = []
		self.id = TreeNode.__id
		TreeNode.__id += 1

	def setParent(self, parent):
		self.parent = parent

	def addChild(self, child):
		if child.id not in self.children:
			self.children.append(child)

	def getChildren(self):
		return self.children

	def getParent(self):
		return self.parent

	def getNeighbors(self):
		neighbors = list(self.children)
		if self.parent:
			neighbors.append(self.parent)
		return neighbors

	def isRoot(self):
		return self.parent == None

	def isSwitch(self):
		return self.attrs.get('isSwitch', False)

	def isHost(self):
		return self.attrs.get('isHost', False)

	def toDotNode(self):
		raise NotImplementedError

	def getReprStr(self, level=None):
		level = 0 if level == None else level
		tab = ''
		for _ in range(level):
			tab += '\t'

		ret = tab
		ret += self.getFullName()
		ret += '\n'
		for child in self.children:
			ret += child.getReprStr(level+1)
		return ret

class Switch(TreeNode):
	def __init__(self, **attrs):
		super(Switch, self).__init__(isSwitch=True, **attrs)

	def getFullName(self):
		return self.getName()

	def getName(self):
		return 's%d' % self.id

	def __repr__(self):
		return self.getName()

	def toDotNode(self):
		return pydot.Node(self.getName(), **self.dotSettings)

class VM(object):
	__id = 0
	dotSettings = {
		'style': 'filled',
		'fillcolor': 'green'
	}
	def __init__(self):
		self.reset()
		self.id = VM.__id
		VM.__id += 1

	@classmethod
	def resetId(cls):
		cls.__id = 0

	def reset(self):
		self.flows = []
		self.host = None

	def addFlow(self, flow):
		if flow not in self.flows:
			self.flows.append(flow)
		flow.vm = self

	def getFlows(self):
		return self.flows

	def setHost(self, host):
		self.host = host

	def getHost(self):
		return self.host

	def getTotalshowFlowCapacity(self):
		return sum(f.amount for f in self.flows)

	def getFullName(self):
		return self.getName()

	def getName(self, capacity=None):
		ret = 'v%d' % self.id
		if capacity:
			ret += '\n(%.2f)' % (self.getTotalshowFlowCapacity())
		return ret

	def __repr__(self):
		return '%s (%r)' % (self.getName(), self.flows)

	def toDotNode(self):
		return pydot.Node(self.getName(capacity=True), **self.dotSettings)

class Flow(object):
	__id = 0
	dotSettings = {
		'style': 'filled',
		'fillcolor': 'blue',
		'shape': 'rectangle'
	}
	def __init__(self, amount, host):
		'''
			A flow is associated with a host, and it will be covered by a VM,
			which may be on another host machine.
		'''
		self.amount = amount
		self.host = host

		self.reset()

		self.id = Flow.__id
		Flow.__id += 1

	def reset(self):
		self.vm = None
		self.hops = [self.host]

	def getName(self, amount=None, capacity=None):
		ret = 'f%d' % (self.id)
		if amount and capacity:
			ret += '\n(%.2f/%.2f)' % (self.getAmount(), self.amount)
		elif amount:
			ret += '\n(%.2f)' % (self.getAmount())
		elif capacity:
			ret += '\n(%.2f)' % (self.amount)
		return ret

	def getFullName(self):
		return 'f%d (%.2f/%.2f)' % (self.id, self.getAmount(), self.amount)

	def setVM(self, vm):
		assert isinstance(vm, VM)
		vm.addFlow(self)
		self.vm = vm

	def getVm(self):
		return self.vm

	def getAmount(self):
		if self.vm:
			return 0
		return self.amount

	def getHost(self):
		return self.host

	def __lt__(self, other):
		return self.amount < other.amount

	def __repr__(self):
		return self.getFullName()

	def toDotNode(self, amount=None, capacity=None):
		return pydot.Node(
			self.getName(capacity=capacity, amount=amount),
			**self.dotSettings)

class Host(TreeNode):
	dotSettings = {
		'style': 'filled',
		'fillcolor': 'red'
	}
	def __init__(self, **attrs):
		super(Host, self).__init__(isHost=True, **attrs)
		self.flows = []
		self.vms = []

	def reset(self):
		for f in self.getFlows():
			f.reset()
		self.vms = []

	def addFlow(self, amount):
		self.flows.append(Flow(amount, self))

	def addVm(self, vm):
		if vm not in self.vms:
			self.vms.append(vm)
			vm.setHost(self)

	def getVms(self):
		return self.vms

	def removeFlow(self, flow):
		self.flows.remove(flow)

	def getTotalFlowAmount(self):
		ret = 0
		for f in self.flows:
			ret += f.getAmount()
		return ret

	def getFlows(self):
		return self.flows

	def getName(self, showFlowCapacity=None):
		ret = 'h%d' % (self.id)
		if showFlowCapacity:
			ret += '\n(%.2f)' % (sum(f.amount for f in self.flows))
		return ret

	def getFullName(self):
		return 'h%d %r' % (self.id, self.flows)

	def __repr__(self):
		return self.getFullName()

	def toDotNode(self):
		return pydot.Node(self.getName(showFlowCapacity=True), **self.dotSettings)

class Tree(object):
	def __init__(self, depth=None, fanout=None):
		if depth == None or depth <= 0 or fanout == None or fanout <= 0:
			self.root = None
			return
		else:
			self.root = Switch(height=0)
			self.nodes = [self.root]
			self.depth = depth

			# build the switches
			self.buildTree(self.root, depth-1, fanout)

			# build the hosts
			for node in self.getNodesAtHeight(self.depth-1):
				for _ in range(fanout):
					h = Host(height=self.depth)
					self.nodes.append(h)
					node.addChild(h)
					h.setParent(node)

	def reset(self):
		# Flow.resetId()
		# VM.resetId()
		for h in [n for n in self.nodes if isinstance(n, Host)]:
			h.reset()
			# for f in h.getFlows():
			# 	f.reset()
			# for vm in h.getVms():
			# 	vm.reset()

	def buildTree(self, node, depth, fanout):
		if depth == 0:
			return
		for _ in range(fanout):
			s = Switch(height=self.depth-depth)
			s.setParent(node)
			node.addChild(s)
			self.nodes.append(s)
			self.buildTree(s, depth-1, fanout)

	def getNodesAtHeight(self, height):
		return [s for s in self.nodes if s.attrs['height'] == height]

	def getHosts(self):
		return [n for n in self.nodes if n.isHost()]

	def getSwitches(self):
		return [n for n in self.nodes if n.isSwitch()]

	def getFlows(self):
		flows = []
		for h in self.getHosts():
			map(flows.append, h.getFlows())
		return flows

	def __repr__(self):
		return self.root.getReprStr(0)

	def getTotalFlowAmount(self):
		amount = 0
		for h in self.getHosts():
			amount += sum(f.amount for f in h.getFlows())
		return amount

	def draw(self, filename, showFlowAmount=None, showFlowCapacity=None, showFlowVm=None):
		'''
			Assuming there is no loop in the tree.
		'''
		g = pydot.Dot(graph_type='graph')
		# g.add_node(pydot.TreeNode(str('alpha = %.2f' % alpha), shape='rectangle'))

		# BFS
		queue = Queue.Queue()
		queue.put(self.root)
		while not queue.empty():
			parent = queue.get()
			parentNode = parent.toDotNode()
			g.add_node(parentNode)
			children = parent.getChildren()
			for child in children:
				queue.put(child)
				childNode = child.toDotNode()
				g.add_node(childNode)
				g.add_edge(pydot.Edge(parentNode, childNode))
				# TODO: refactor into a function
				if isinstance(child, Host):
					for flow in child.getFlows():
						flowNode = flow.toDotNode(
							amount=showFlowAmount,
							capacity=showFlowCapacity)
						g.add_node(flowNode)
						g.add_edge(pydot.Edge(childNode, flowNode))

						if showFlowVm:
							vmNode = flow.getVm().toDotNode()
							g.add_node(vmNode)
							g.add_edge(pydot.Edge(flowNode, vmNode, color='blue'))

					for vm in child.getVms():
						g.add_edge(pydot.Edge(childNode, vm.toDotNode(), color='red'))

		g.write_png(filename)

class Solver(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.hostToVms = {}
		self.switchToFlows = {}

	def solve(self, tree):
		raise NotImplementedError

	@classmethod
	def dist(cls, node1, node2):
		'''
			Returns the distance between two nodes (an integer).
		'''
		assert isinstance(node1, TreeNode), '%r is not a node' % node1
		assert isinstance(node2, TreeNode), '%r is not a node' % node2
		discovered = {}
		queue = Queue.Queue()
		queue.put((node1, 0))
		# discovered[node1] = True
		# neighbors = node1.getNeighbors()
		# map(queue.put, [(neighbor, 1) for neighbor in neighbors])
		# discovered.update(dict([(neighbor, True) for neighbor in neighbors]))
		while not queue.empty():
			node, distance = queue.get()
			if node == node2:
				return distance
			elif discovered.get(node, False):
				continue
			else:
				neighbors = node.getNeighbors()
				map(queue.put, [(neighbor, distance + 1) for neighbor in neighbors])
				discovered[node] = True
				# discovered.update(dict([(neighbor, True) for neighbor in neighbors]))
		raise Exception('Unable to find distance between %r and %r' % (node1, node2))

	def getSwitchTotalFlowAmount(self, switch):
		self.switchToFlows.setdefault(switch, [])
		flows = self.switchToFlows[switch]
		amount = 0
		for f in flows:
			amount += f.getAmount()
		return amount

	def pushResidualFlowsToParent(self, node):
		parent = node.getParent()
		assert parent.isSwitch()
		self.switchToFlows.setdefault(parent, [])

		if node.isHost():
			for f in node.getFlows():
				if f.getAmount() > 0:
					self.switchToFlows[parent].append(f)
					f.hops.append(parent)
		else:
			for f in self.switchToFlows[node]:
				if f.getAmount() > 0:
					self.switchToFlows[parent].append(f)
					f.hops.append(parent)

	def coverHostWithVm(self, host, forced=False):
		raise NotImplementedError

	def findBestHostToCoverSwitch(self, switch):
		'''
			Greedily finds the path with maximum aggregate flow to the bottom (hosts).
			NOTE: we assume there is NO LOOP in a flow's path (hops)
			@return a host
		'''
		# assume there is no loop in a flow's path (hops)
		# FIXME: [0.6, 0.6] will be in a vnf
		# NOTE: we should "get" the "host" to deploy the VNF, and cover the flow
		# in the top level function (get the "host" recursively, deploy once)
		# Use first-fit algorithm to deploy (deploy on VNF at a time)
		maxFlowAmount = 0
		maxFlows = []
		maxFlowNode = None
		for child in switch.getChildren():
			flows = [f for f in self.switchToFlows[switch] if child in f.hops]
			# for f in self.switchToFlows[switch]:
			# 	print f.hops
			# print 'child: ', child, flows
			amount = sum(f.getAmount() for f in flows)
			if amount > maxFlowAmount:
				maxFlowAmount = amount
				maxFlows = flows
				maxFlowNode = child

		# print maxFlowAmount, maxFlows, maxFlowNode

		if isinstance(maxFlowNode, Host):
			return maxFlowNode
		else:
			return self.findBestHostToCoverSwitch(maxFlowNode)

	def coverSwitchWithVm(self, switch):
		'''
			Cover a maximum set of flow on `switch` by deploying a VM on the target
			host found by `findBestHostToCoverSwitch`.
		'''
		# Note: this function must be called before the flows are added to a VM
		host = self.findBestHostToCoverSwitch(switch)
		assert host is not None

		flows = self.findMaximumFlowSetOnSwitch(switch)
		vm = VM()
		for f in flows:
			vm.addFlow(f)

		self.hostToVms.setdefault(host, [])
		self.hostToVms[host].append(vm)
		host.addVm(vm)

	def findMaximumFlowSetOnSwitch(self, switch):
		amount = 0
		flows = []
		for f in sorted(self.switchToFlows[switch], reverse=True):
			if amount + f.getAmount() > 1:
				break
			if f.getAmount() > 0:
				amount += f.getAmount()
				flows.append(f)
		return flows

	def findMaximumFlowSetOnFlows(self, flows):
		amount = 0
		maxFlowSet = []
		for f in sorted(flows, reverse=True):
			if amount + f.getAmount() > 1:
				continue
			if f.getAmount() > 0:
				amount += f.getAmount()
				maxFlowSet.append(f)
		return maxFlowSet

	def getSolution(self, showVmPlacement=True, showSummary=True):
		ret = ''
		if showVmPlacement:
			for h in self.hostToVms.keys():
				ret += 'host-%d' % h.id
				ret += '\n\t'
				ret += '%r\n' % self.hostToVms[h]

		if showSummary:
			ret += self.getSummary()
		return ret

	def getTotalCost(self):
		'''
			Returns the total cost caused by the flows.
			Cost = sum of (flowAmount * distFromVmToHost) for each flow
		'''
		totalCost = 0
		for vms in self.hostToVms.values():
			for vm in vms:
				for f in vm.getFlows():
					totalCost += f.amount * self.dist(f.getHost(), f.getVm().getHost())
		return totalCost

	def getSummary(self):
		ret = 'Number of VMs: %d' % (self.getNumVms())
		ret += '\nTotal Traffic Cost: %.2f' % (self.getTotalCost())
		return ret + '\nTotal Amount of Flows: %.2f' % (self.getTotalFlowAmount())

	def getNumVms(self):
		numVms = 0
		for h in self.hostToVms.keys():
			numVms += len(self.hostToVms[h])
		return numVms

	def getTotalFlowAmount(self):
		'''
			Returns the aggregate flow amount on all flows.
		'''
		totalFlowAmount = 0
		for vms in self.hostToVms.values():
			for vm in vms:
				totalFlowAmount += sum(f.amount for f in vm.getFlows())
		return totalFlowAmount

class AlphaSolver(Solver):
	def __init__(self, alpha=0.8):
		super(AlphaSolver, self).__init__()
		self.alpha = alpha

	def solve(self, tree):
		assert isinstance(tree, Tree)
		self.reset()

		for h in tree.getHosts():
			while h.getTotalFlowAmount() >= self.alpha and self.coverHostWithVm(h):
				pass
			self.pushResidualFlowsToParent(h)

		# bottom up
		for height in reversed(range(tree.depth)):
			for s in tree.getNodesAtHeight(height):
				while self.getSwitchTotalFlowAmount(s) >= self.alpha:
					# print self.getSwitchTotalFlowAmount(s)
					self.coverSwitchWithVm(s)
				if not s.isRoot():
					self.pushResidualFlowsToParent(s)

		if self.getSwitchTotalFlowAmount(tree.root) > 0:
			self.coverSwitchWithVm(tree.root)

	def coverHostWithVm(self, host, forced=False):
		'''
			Covers a host with a VM using first-fit algorithm.
			If `forced=True`, a VM will be deployed under any circumstance.
		'''
		vm = VM()
		self.hostToVms.setdefault(host, [])
		amount = 0
		flows = []
		for f in sorted(host.getFlows(), reverse=True):
			if f.getAmount() == 0:
				continue
			if amount + f.getAmount() > 1:
				break
			amount += f.getAmount()
			flows.append(f)

		if (amount >= self.alpha and amount <= 1) or forced:
			for f in flows:
				vm.addFlow(f)
			self.hostToVms[host].append(vm)
			host.addVm(vm)
			return True
		else:
			return False

	def getSummary(self):
		ret = 'Alpha: %.2f\n' % (self.alpha)
		return ret + super(AlphaSolver, self).getSummary()

	def getName(self):
		return 'Alphasolver (%.2f)' % self.alpha

class LeastVnfSolver(Solver):
	def __init__(self):
		super(LeastVnfSolver, self).__init__()

	def solve(self, tree):
		assert isinstance(tree, Tree)
		discoveredNodes = []
		queue = Queue.Queue()
		map(queue.put, tree.getHosts())
		while not queue.empty():
			node = queue.get()
			if node.isRoot():
				break
			self.pushResidualFlowsToParent(node)
			if node.getParent() not in discoveredNodes:
				queue.put(node.getParent())
				discoveredNodes.append(node.getParent())

		targetHost = self.findBestHostToCoverSwitch(tree.root)
		self.hostToVms.setdefault(targetHost, [])
		# first-fit
		flows = []
		for host in tree.getHosts():
			flows.extend(host.getFlows())

		# TODO: Improve the efficiency
		# (findMaximumFlowSetOnFlows will iterate through all flows every time)
		# O(n^2) -> O(n)
		while True:
			maxFlowSet = self.findMaximumFlowSetOnFlows(flows)
			map(flows.remove, maxFlowSet)
			if len(maxFlowSet) == 0:
				break
			vm = VM()
			map(vm.addFlow, maxFlowSet)
			self.hostToVms[targetHost].append(vm)
			targetHost.addVm(vm)

	def getName(self):
		return 'LeastVnfSolver'


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
		flowAmounts = [f.amount for f in allFlows]
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
			solver.solve(self.tree)
			if drawPng:
				self.tree.draw(
					'%s/%s.png (%s)' % (pngOutputDir, self.name, solver.getName()),
					showFlowCapacity=True,
					showFlowVm=True)
			records.append({
				'solver': solver.getName(),
				'numVms': solver.getNumVms(),
				'cost': solver.getTotalCost()
			})
			# print solver.getSolution(
			# 	showVmPlacement=printVmPlacement,
			# 	showSummary=printSummary)
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

if __name__ == '__main__':
	proposedSolver = AlphaSolver(alpha=0.8)
	leastCostSolver = AlphaSolver(alpha=0.0001)
	leastVnfSolver = LeastVnfSolver()

	numFlowsPerHostSettings = [
		(1, 5)
		# (5, 10),
		# (10, 50),
		# (100, 500),
		# (1000, 5000)
		# (10000, 50000)
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
