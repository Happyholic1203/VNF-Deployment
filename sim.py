class Node(object):
	__id = 0
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

	def getReprStr(self, level=None):
		level = 0 if level == None else level
		tab = ''
		for _ in range(level):
			tab += '\t'

		ret = tab
		ret += self.getName()
		ret += '\n'
		for child in self.children:
			ret += child.getReprStr(level+1)
		return ret

class Switch(Node):
	def __init__(self, **attrs):
		super(Switch, self).__init__(isSwitch=True, **attrs)

	def getName(self):
		return 's%d' % self.id

	def __repr__(self):
		return self.getName()

class VM(object):
	__id = 0
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

	def getName(self):
		return 'v%d' % self.id

	def __repr__(self):
		return '%s (%r)' % (self.getName(), self.flows)

class Flow(object):
	__id = 0
	def __init__(self, amount, host):
		self.amount = amount
		self.vm = None
		self.host = host
		self.id = Flow.__id
		Flow.__id += 1

	def getName(self):
		# return 'flow-%d (%.2f)' % (self.id, self.getAmount())
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

	def __repr__(self):
		return self.getName()

class Host(Node):
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
		return 'h%d %r' % (self.id, self.flows)

	def __repr__(self):
		return self.getName()

class Tree(object):
	def __init__(self, depth=None, fanout=None):
		if depth == None or depth <= 0 or fanout == None or fanout <= 0:
			self.root = None
			return
		else:
			self.root = Switch(height=0)
			self.nodes = [self.root]
			self.depth = depth
			self.buildTree(self.root, depth-1, fanout)
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

class Solver(object):
	def __init__(self, alpha=0.8):
		self.alpha = alpha
		self.reset()

	def reset(self):
		self.hostToVms = {}
		self.switchToFlows = {}

	def solve(self, tree):
		assert isinstance(tree, Tree)
		self.reset()

		for h in tree.getHosts():
			while h.getTotalFlowAmount() >= self.alpha:
				self.coverHostWithVm(h)
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
		else:
			for f in self.switchToFlows[node]:
				if f.getAmount() > 0:
					self.switchToFlows[parent].append(f)


	# TODO: improve this function to provide optimal solution
	# Flows can add up in many combinations, and one of them is closest
	# to the alpha value. This function should find the closest flow
	# combination that best matches the alpha value.
	def coverHostWithVm(self, host):
		vm = VM()
		self.hostToVms.setdefault(host, [])
		amount = 0
		for f in host.getFlows():
			if amount + f.getAmount() > 1:
				break
			amount += f.getAmount()
			vm.addFlow(f)
		self.hostToVms[host].append(vm)

	def coverSwitchWithVm(self, switch):
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
		return ret


if __name__ == '__main__':
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

	hosts[5].addFlow(0.45)
	hosts[5].addFlow(0.5)

	hosts[6].addFlow(0.1)
	hosts[6].addFlow(0.1)
	hosts[6].addFlow(0.2)

	hosts[7].addFlow(0.3)
	hosts[7].addFlow(0.3)
	hosts[7].addFlow(0.2)

	print '*** Tree before solving:\n', t

	s = Solver(alpha=0.8)
	s.solve(t)

	print '*** Tree after solving:\n', t

	print '*** VMs under each host:\n', s.getSolution()