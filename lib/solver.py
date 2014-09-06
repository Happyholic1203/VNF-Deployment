from ds import *
import Queue
import time

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
		maxFlowAmount = 0
		maxFlows = []
		maxFlowNode = None
		for child in switch.getChildren():
			flows = [f for f in self.switchToFlows[switch] if child in f.hops]
			amount = sum(f.getAmount() for f in flows)
			if amount > maxFlowAmount:
				maxFlowAmount = amount
				maxFlows = flows
				maxFlowNode = child

		assert maxFlowNode is not None

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
		sortedFlows = sorted(flows, reverse=True)
		for f in sortedFlows:
			if amount + f.getAmount() > 1:
				# early stop for better efficiency
				if amount + sortedFlows[-1].getAmount() > 1:
					break
				else:
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
					totalCost += f.getCapacity() * self.dist(f.getHost(), f.getVm().getHost())
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
				totalFlowAmount += sum(f.getCapacity() for f in vm.getFlows())
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
