import pydot
import Queue

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
		flow.setVM(self)

	def getFlows(self):
		return self.flows

	def setHost(self, host):
		self.host = host

	def getHost(self):
		return self.host

	def getTotalshowFlowCapacity(self):
		return sum(f.getCapacity() for f in self.flows)

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
		self.capacity = amount
		self.host = host

		self.reset()

		self.id = Flow.__id
		Flow.__id += 1

	def reset(self):
		self.vm = None
		self.amount = self.capacity
		self.hops = [self.host]

	def getName(self, amount=None, capacity=None):
		ret = 'f%d' % (self.id)
		if amount and capacity:
			ret += '\n(%.2f/%.2f)' % (self.getAmount(), self.getCapacity())
		elif amount:
			ret += '\n(%.2f)' % (self.getAmount())
		elif capacity:
			ret += '\n(%.2f)' % (self.getCapacity())
		return ret

	def getFullName(self):
		return 'f%d (%.2f/%.2f)' % (self.id, self.getAmount(), self.getCapacity())

	def setVM(self, vm):
		assert isinstance(vm, VM)
		self.vm = vm
		self.amount = 0

	def getVm(self):
		return self.vm

	def getAmount(self):
		return self.amount

	def getCapacity(self):
		return self.capacity

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
			ret += '\n(%.2f)' % (sum(f.getCapacity() for f in self.flows))
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
			amount += sum(f.getCapacity() for f in h.getFlows())
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
