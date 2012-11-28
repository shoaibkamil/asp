import ast

"""
I don't use the Generable class inheritance
"""

class Generable():
	pass

class func_types(Generable):	
	def __init__(self, types):
		self.types = types
		self._fields = []			
	
class Number(Generable):
	def __init__(self, num):
		self.num = num
		self._fields = []
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self

class String(Generable):
	def __init__(self, text):
		self.text = text
		self._fields = ['text']
		self.done = False
	
	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self

class Name(Generable):
	def __init__(self,name):
		self.name= name
		self._fields= []
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self

class Function(Generable):
	def __init__(self, declaration, body):
		self.declaration = declaration
		self.body = body
		self._fields = []
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self

class Arguments(Generable):
	def __init__(self, args):
		self.args = args
		self._fields = []
		
class FunctionDeclaration(Generable):
	def __init__(self, name, args):
		self.name = name
		self.args = args

class Expression(Generable):
	def __init__(self):
		# ???
		super(Expression, self)
		self._fields = []
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self

class Call(Expression):	
	def __init__(self, func, args):
		self.func = func
		self.args = args
		self._fields = []
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self
	
class Attribute(Expression):	
	def __init__(self, value, attr):	
		self.attr = attr
		self.value = value
		
class List(Expression):
	def __init__(self, elements):
		self.elements = elements
		self._fields = []
		
class BinOp(Expression):
	def __init__(self, left, op, right):
		self. left = left
		self.op = op
		self.right = right
		self._fields = ['left', 'right']
		self.done = False

class BoolOp(Expression):
	def __init__(self, op, values):
		self.op = op
		self.values = values
		self._fields = ['op', 'values']
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self

class UnaryOp(Expression):
	def __init__(self, op, operand):
		self.op = op
		self.operand = operand
		self._fields = ['operand']	

class Subscript(Expression):
	def __init__(self, value, index, context):
		self.value = value
		self.index = index
		self.context = context
		self._fields = ['value', 'index', 'context']

class Print(Generable):
	def __init__(self,text,newline,dest):
		self.text = text
		self.newline = newline
		self.dest= dest
		self.done = False
		
	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self
		
class ReturnStatement(Generable):
	def __init__(self, retval):
		self.retval = retval
		self._fields = ['retval']
		self.done = False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self
		
class AugAssign(Generable):
	def __init__(self, target, op, value):
		self.target = target
		self.op = op
		self.value = value
		self.done = False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self
		

class Assign(Generable): #should this inherit from something else??
	def __init__(self, lvalue, rvalue):
		##??
		self.lvalue = lvalue
		self.rvalue= rvalue
		self._fields = ['lvalue', 'rvalue']
		self.done = False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self
		
		
class Compare(Generable):
	def __init__(self, left,op,right):
		self.left = left
		self.op = op
		self.right = right
		self.done=False
		self._fields = ('left', 'op', 'right')
		
	def __iter__(self):	
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done=True
			return self
		
class IfConv(Generable):
	def __init__(self, test, body, orelse, inner_if=False):
		self.test = test
		self.body = body
		self.orelse = orelse
		self.inner_if = inner_if
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self
		
class For(Generable): 
	def __init__(self, target, iter, body):
		self.target = target
		self.iter = iter
		self.body = body
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self

class While(Generable):
	def __init__(self, test, body):
		self.test = test
		self.body = body
		self._fields = []
		self.done= False

	def __iter__(self):
		return self
	
	def next(self):
		if self.done:
			raise StopIteration
		else:
			self.done = True
			return self
	
	
if __name__ == '__main__':
	pass

