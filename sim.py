import pydot
import Queue

class Node(object):
	__id = 0
	dotSettings = {
		'style': 'filled',
		'fillcolor': 'white'
	}
	def __init__(self, **attrs):
		self.attrs = attrs
		self.parent = None
		self.children = []
		self.id = Node.__id
		Node.__id += 1

	def setParent(self, parent):
		self.parent = parent

	def addChild(self, child):
		if child.id not in self.children:
			self.children.append(child)

	def getChildren(self):
		return self.children

	def getParent(self):
		return self.parent

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

class Switch(Node):
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
		self.flows = []
		self.id = VM.__id
		VM.__id += 1

	def addFlow(self, flow):
		if flow not in self.flows:
			self.flows.append(flow)
		flow.vm = self

	def getFlows(self):
		return self.flows

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
		self.amount = amount
		self.vm = None
		self.host = host
		self.hops = [host]
		self.id = Flow.__id
		Flow.__id += 1

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

	def getVM(self):
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

class Host(Node):
	dotSettings = {
		'style': 'filled',
		'fillcolor': 'red'
	}
	def __init__(self, **attrs):
		super(Host, self).__init__(isHost=True, **attrs)
		self.flows = []

	def addFlow(self, amount):
		self.flows.append(Flow(amount, self))

	def removeFlow(self, flow):
		self.flows.remove(flow)

	def getTotalFlowAmount(self):
		ret = 0
		for f in self.flows:
			ret += f.getAmount()
		return ret

	def getFlows(self):
		return self.flows

	def getName(self):
		return 'h%d' % (self.id)

	def getFullName(self):
		return 'h%d %r' % (self.id, self.flows)

	def __repr__(self):
		return self.getFullName()

	def toDotNode(self):
		return pydot.Node(self.getName(), **self.dotSettings)

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

	def __repr__(self):
		return self.root.getReprStr(0)

	def draw(self, filename, showFlowAmount=None, showFlowCapacity=None, showFlowVm=None):
		'''
			Assuming there is no loop in the tree.
		'''
		g = pydot.Dot(graph_type='graph')

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
				if isinstance(child, Host):
					for flow in child.getFlows():
						flowNode = flow.toDotNode(
							amount=showFlowAmount,
							capacity=showFlowCapacity)
						g.add_node(flowNode)
						g.add_edge(pydot.Edge(childNode, flowNode))

						if showFlowVm:
							vmNode = flow.getVM().toDotNode()
							g.add_node(vmNode)
							g.add_edge(pydot.Edge(flowNode, vmNode))

		g.write_png(filename)

class Solver(object):
	def __init__(self, alpha=0.8):
		self.alpha = alpha
		self.reset()

	def reset(self):
		self.hostToVms = {}
		self.switchToFlows = {}

	def solve(self, tree, noPrint=None):
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

		if not noPrint:
			print 'Number of VMs: %d' % (self.getNumVms()) + \
				'\nTotal Amount of Flows: %f' % (self.getTotalFlowAmount())

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
					# print 'pushing %s onto %s.hops' % (parent, f)
					f.hops.append(parent)
		else:
			for f in self.switchToFlows[node]:
				if f.getAmount() > 0:
					self.switchToFlows[parent].append(f)
					f.hops.append(parent)

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
			if amount + f.getAmount() > 1:
				break
			amount += f.getAmount()
			flows.append(f)

		if amount >= self.alpha or forced:
			for f in flows:
				vm.addFlow(f)
			self.hostToVms[host].append(vm)
			return True
		else:
			return False

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


	def coverSwitchWithVmOld(self, switch):
		vm = VM()

		maxAmount = -1
		maxFlow = None
		for f in self.switchToFlows[switch]:
			if f.getAmount() > maxAmount:
				maxFlow = f
		amount = maxFlow.getAmount()

		# make sure that the maxFlow is covered by the VM
		vm.addFlow(maxFlow)
		host = maxFlow.getHost()
		self.hostToVms.setdefault(host, [])

		for f in self.switchToFlows[switch]:
			if f == maxFlow:
				continue
			if amount + f.getAmount() > 1:
				break
			amount += f.getAmount()
			vm.addFlow(f)

		self.hostToVms[host].append(vm)

	def getSolution(self):
		ret = ''
		for h in self.hostToVms.keys():
			ret += 'host-%d' % h.id
			ret += '\n\t'
			ret += '%r\n' % self.hostToVms[h]

		ret += '\nNumber of VMs: %d' % (self.getNumVms())
		return ret + '\nTotal Amount of Flows: %f' % (self.getTotalFlowAmount())

	def getNumVms(self):
		numVms = 0
		for h in self.hostToVms.keys():
			numVms += len(self.hostToVms[h])
		return numVms

	def getTotalFlowAmount(self):
		totalFlowAmount = 0
		for vms in self.hostToVms.values():
			for vm in vms:
				totalFlowAmount += sum(f.amount for f in vm.getFlows())
		return totalFlowAmount


if __name__ == '__main__':
	s = Solver(alpha=0.8)

	# 2^3 = 8 hosts
	t = Tree(depth=3, fanout=2)
	hosts = t.getHosts()
	hosts[0].addFlow(0.2)
	hosts[0].addFlow(0.1)
	hosts[0].addFlow(0.1)
	hosts[0].addFlow(0.1)

	hosts[1].addFlow(0.2)
	hosts[1].addFlow(0.3)
	hosts[1].addFlow(0.2)

	hosts[2].addFlow(0.1)
	hosts[2].addFlow(0.5)
	hosts[2].addFlow(0.2)

	hosts[3].addFlow(0.3)
	hosts[3].addFlow(0.3)
	hosts[3].addFlow(0.1)

	hosts[4].addFlow(0.25)
	hosts[4].addFlow(0.25)
	hosts[4].addFlow(0.25)

	hosts[5].addFlow(0.6)
	hosts[5].addFlow(0.6)

	hosts[6].addFlow(0.1)
	hosts[6].addFlow(0.1)
	hosts[6].addFlow(0.2)

	hosts[7].addFlow(0.3)
	hosts[7].addFlow(0.3)
	hosts[7].addFlow(0.2)

	t.draw('topo.png', showFlowCapacity=True)
	s.solve(t)

	t.draw('topo_solved.png', showFlowCapacity=True, showFlowVm=True)
