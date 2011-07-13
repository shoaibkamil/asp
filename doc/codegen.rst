:mod:`asp.codegen` -- Code Generation API Reference
===================================================

.. module:: asp.codegen

This module contains the Python and C++ AST modules as well
as tools for traversal and manipulation of the AST.


:mod:`asp.codegen.ast_tools` -- AST Manipulation/Traversal Tools
----------------------------------------------------------------

.. automodule:: asp.codegen.ast_tools

.. autoclass:: NodeVisitor
   :members:
   :inherited-members:

   Subclass of the Python AST NodeVisitor class. See the `ast.NodeVisitor
   <http://docs.python.org/library/ast.html#ast.NodeVisitor>`_ documentation.

.. autoclass:: NodeTransformer
   :members:
   :inherited-members:   
   
   Subclass of the Python AST NodeTransformer class.  See the
   `ast.NodeTransformer
   <http://docs.python.org/library/ast.html#ast.NodeTransformer>`_
   documentation.

.. autoclass:: ASTNodeReplacer
   :members:
   :inherited-members:

.. autoclass:: ConvertAST
   :members:
   :inherited-members:

.. autoclass:: LoopUnroller
   :members:
   :inherited-members:


:mod:`asp.codegen.cpp_ast` -- C++ AST 
-------------------------------------

.. automodule:: asp.codegen.cpp_ast

.. autoclass:: CNumber
   :members:

.. autoclass:: String
   :members:

.. autoclass:: CName
   :members:

.. autoclass:: Expression
   :members:

.. autoclass:: BinOp
   :members:

.. autoclass:: UnaryOp
   :members:

.. autoclass:: Subscript
   :members:

.. autoclass:: Call
   :members:

.. autoclass:: PostfixUnaryOp
   :members:

.. autoclass:: ConditionalExpr
   :members:

.. autoclass:: TypeCast
   :members:

.. autoclass:: ForInitializer
   :members:

.. autoclass:: RawFor
   :members:

.. autoclass:: For
   :members:

.. autoclass:: FunctionBody
   :members:

.. autoclass:: FunctionDeclaration
   :members:


.. autoclass:: Value
   :members:


.. autoclass:: Pointer
   :members:


.. autoclass:: Block
   :members:


.. autoclass:: UnbracedBlock
   :members:


.. autoclass:: Define
   :members:


.. autoclass:: Statement
   :members:


.. autoclass:: Assign
   :members:


.. autoclass:: FunctionCall
   :members:


.. autoclass:: Print
   :members:


:mod:`asp.codegen.python_ast` -- Python AST
-------------------------------------------

.. automodule:: asp.codegen.python_ast

   See the `Python AST Docs <http://docs.python.org/library/ast.html>`_ .
