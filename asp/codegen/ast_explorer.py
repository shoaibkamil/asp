#!/usr/bin/env python

# Based on example basictreeview.py from:
# http://www.pygtk.org/pygtk2tutorial/examples/basictreeview.py

import pygtk
pygtk.require('2.0')
import gtk
import inspect
import re
from types import *

def debug_str(obj):
    if isinstance(obj, str):
        # TODO: needs escaping
        return "'" + obj + "'"
    elif isinstance(obj, list):
        return '[' + ', '.join([debug_str(x) for x in obj]) + ']'
    else:
        result = str(obj)
        result = re.sub(r'<_ast\.(.*) object at 0x[0-9a-f]+>', r'\1', result)
        return result

def generator_index(gen, index):
    import itertools
    return next(itertools.islice(gen, index, index+1))

class ASTExplorer:

    def button_release_event(self, treeview, event):
        if event.button == 3: # right click
            result = treeview.get_path_at_pos(int(event.x), int(event.y))
            if result != None:
                self.path_right_clicked = result[0]
                self.context_menu.popup(None, None, None, event.button, event.time, None)
                self.context_menu.show_all()

    def copy_expression(self, menuitem, event):
        if event.button == 1: # left click
            path = 'ast' + self.get_path(self.tree, self.path_right_clicked[1:])
            self.clipboard.set_text(path)
            self.path_right_clicked = None

    def reduced_pairs(self, obj):
        # Exclude lineno, col_offset (from Python AST), _fields
        # (from CodePy C++ AST) to simplify tree
        dict = obj.__dict__
        dict.pop('lineno', None)
        dict.pop('col_offset', None)
        dict.pop('_fields', None)
        return [(key, dict[key]) for key in sorted(dict.iterkeys())]

    def get_path(self, tree, path):
        if len(path) == 0:
            return ''
        key, value = generator_index(self.reduced_pairs(tree), path[0])
        result = '.' + key
        path = path[1:]
        if len(path) == 0:
            return result
        if isinstance(value, list):
            result += '[' + str(path[0]) + ']'
            value = value[path[0]]
            path = path[1:]
        return result + self.get_path(value, path)

    def add_tree(self, tree, parent):
        if isinstance(tree, (int, str, NoneType)):
            return
        for key, value in self.reduced_pairs(tree):
            attriter = self.treestore.append(parent, [key + ': ' + debug_str(value)])
            if isinstance(value, list):
                for i in range(0, len(value)):
                    elemiter = self.treestore.append(attriter, ['[' + str(i) + '] ' + debug_str(value[i])])
                    self.add_tree(value[i], elemiter)
            else:
                self.add_tree(value, attriter)

    # close the window and quit
    def delete_event(self, widget, event, data=None):
        gtk.main_quit()
        return False

    def __init__(self, tree):
        self.tree = tree

        # Create a new window
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title("AST Explorer")
        self.window.set_size_request(400, 500)
        self.window.connect("delete_event", self.delete_event)

        self.clipboard = gtk.clipboard_get(gtk.gdk.SELECTION_CLIPBOARD)

        # Create right-click context menu
        self.context_menu = gtk.Menu()
        menuitem_copyexpr = gtk.MenuItem("Copy expression")
        menuitem_copyexpr.connect('button-release-event' , self.copy_expression)
        self.context_menu.append(menuitem_copyexpr)

        # create a TreeStore with one string column to use as the
        # model and fill with data
        self.treestore = gtk.TreeStore(str)
        rootiter = self.treestore.append(None, [debug_str(self.tree)])
        self.add_tree(self.tree, rootiter)

        # create the TreeView using treestore and hook up button event
        self.treeview = gtk.TreeView(self.treestore)
        self.treeview.connect('button-release-event' , self.button_release_event)
        self.path_right_clicked = None

        # create the TreeViewColumn to display the data
        self.tvcolumn = gtk.TreeViewColumn()

        # add tvcolumn to treeview
        self.treeview.append_column(self.tvcolumn)

        # create a CellRendererText to render the data
        self.cell = gtk.CellRendererText()

        # add the cell to the tvcolumn and allow it to expand
        self.tvcolumn.pack_start(self.cell, True)

        # set the cell "text" attribute to column 0 - retrieve text
        # from that column in treestore
        self.tvcolumn.add_attribute(self.cell, 'text', 0)

        # make it searchable
        self.treeview.set_search_column(0)

        self.scrolled_window = gtk.ScrolledWindow()
        self.scrolled_window.add(self.treeview)
        self.window.add(self.scrolled_window)
        self.window.show_all()

        gtk.main()

class TestObject:
    def operation(self, x):
        return 2*x+5

if __name__ == "__main__":
    import ast_tools
    ast = ast_tools.parse_method(TestObject.operation)
    ASTExplorer(ast)
