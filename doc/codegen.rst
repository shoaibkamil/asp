:mod:`asp.codegen` -- Code Generation API Reference
===================================================

.. module:: asp.codegen

This module contains the Python and C++ AST modules as well
as tools for traversal and manipulation of the AST.


:mod:`asp.codegen.ast_tools` -- AST Manipulation/Traversal Tools
----------------------------------------------------------------

.. automodule:: asp.codegen.ast_tools

.. class:: NodeVisitor(ast.NodeVisitor)

   Unified class for visiting Python and C++ AST nodes. Subclass
   of the Python AST NodeVisitor class. See the `ast.NodeVisitor
   <http://docs.python.org/library/ast.html#ast.NodeVisitor>`_ documentation.

.. class:: NodeTransformer(ast.NodeTransformer)

   Unified class for transforming Python and C++ AST nodes.  Subclass
   of the Python AST NodeTransformer class.  See the
   `ast.NodeTransformer
   <http://docs.python.org/library/ast.html#ast.NodeTransformer>`_
   documentation.

.. autoclass:: NodeTransformer
   :members: visit, generic_visit
   
   Unified class...
