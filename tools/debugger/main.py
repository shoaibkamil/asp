#!/usr/bin/python
# -*- coding: utf-8 -*-
 
# Based on http://developer.qt.nokia.com/wiki/PySideTutorials_Simple_Dialog

import cgi
import sys
import gdb
import pdb
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtDeclarative import QDeclarativeView

class Form(QDialog):
    def __init__(self, parent=None):
        self.fontFamily = "Courier"
        super(Form, self).__init__(parent)
        self.setWindowTitle("My Form")
        self.resize(1024, 600)
        self.textedit = QTextEdit()
        self.textedit.setLineWrapMode(QTextEdit.NoWrap)
        self.button_step_over = QPushButton("Step over")
        self.button_step_over.clicked.connect(self.stepOver)
        self.button_restart = QPushButton("Restart")
        self.button_restart.clicked.connect(self.restart)
        self.button_close = QPushButton("Close")
        self.button_close.clicked.connect(exit)

        watches = [('C++','x1'), ('C++', 'x2'), ('C++', 'x3'), ('C++', '_my_out_grid[x3]'),
                   ('Python', 'x'), ('Python', 'y'), ('Python', 'out_grid[x]'), ('Python', 'in_grid[y]')]

        self.watch_layout = QGridLayout()
        self.watch_values = dict()
        row = 0
        for watch in watches:
            self.watch_values[watch] = QLineEdit()
            self.watch_layout.addWidget(QLabel(watch[1]), row, 0)
            self.watch_layout.addWidget(self.watch_values[watch], row, 1)
            self.watch_layout.addWidget(QLabel(watch[0]), row, 2)
            row += 1

        self.right_panel = QVBoxLayout()
        self.right_panel.addLayout(self.watch_layout)
        self.right_panel.addStretch()

        self.button_hlayout = QHBoxLayout()
        self.button_hlayout.addWidget(self.button_step_over)
        self.button_hlayout.addWidget(self.button_restart)
        self.button_hlayout.addWidget(self.button_close)

        self.edit_watch_hlayout = QHBoxLayout()
        self.edit_watch_hlayout.addWidget(self.textedit)
        self.edit_watch_hlayout.addLayout(self.right_panel)
        self.edit_watch_hlayout.setStretch(0, 1)

        self.vlayout = QVBoxLayout()
        self.vlayout.addLayout(self.edit_watch_hlayout)
        self.vlayout.addLayout(self.button_hlayout)

        self.setLayout(self.vlayout)
        self.mixedText = """\
from stencil_kernel import *
import stencil_grid
import numpy

class ExampleKernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
@void kernel(PyObject *in_img, PyObject *out_grid)
@{
@  #define _in_img_array_macro(_d0,_d1) (_d1+(_d0 * 10))
@  #define _out_grid_array_macro(_d0,_d1) (_d1+(_d0 * 10))
@  npy_double *_my_in_img = ((npy_double *)PyArray_DATA(in_img));
@  npy_double *_my_out_grid = ((npy_double *)PyArray_DATA(out_grid));
        for x in out_grid.interior_points():
@  for (int x1 = 1; (x1 <= 8); x1 = (x1 + 1))
@  {
@    #pragma IVDEP
@    for (int x2 = 1; (x2 <= 8); x2 = (x2 + 1))
@    {
@      int x3;
@      x3 = _out_grid_array_macro(x1, x2);
            for y in in_grid.neighbors(x, 1):
                out_grid[x] = out_grid[x] + in_grid[y]
@      _my_out_grid[x3] = (_my_out_grid[x3] + _my_in_img[_in_img_array_macro((x1 + 1), (x2 + 0))]);
@      _my_out_grid[x3] = (_my_out_grid[x3] + _my_in_img[_in_img_array_macro((x1 + -1), (x2 + 0))]);
@      _my_out_grid[x3] = (_my_out_grid[x3] + _my_in_img[_in_img_array_macro((x1 + 0), (x2 + 1))]);
@      _my_out_grid[x3] = (_my_out_grid[x3] + _my_in_img[_in_img_array_macro((x1 + 0), (x2 + -1))]);
@    }
@  }
@}

in_grid = StencilGrid([5,5])
in_grid.data = numpy.ones([5,5])

out_grid = StencilGrid([5,5])
ExampleKernel().kernel(in_grid, out_grid)
"""
        self.gdb = gdb.gdb()
        self.pdb = pdb.pdb()
        self.updateView()

    def get_line_from_cpp_line(self, cpp_line):
        current_line = 0
        current_cpp_line = 0
        for line in self.mixedText.split("\n"):
            if len(line) > 1 and line[0] == '@':
                if current_cpp_line == cpp_line:
                    return current_line
                current_cpp_line += 1
            current_line += 1
        return -1

    def get_line_from_python_line(self, python_line):
        current_line = 0
        current_python_line = 0
        for line in self.mixedText.split("\n"):
            if not (len(line) > 1 and line[0] == '@'):
                if current_python_line == python_line:
                    return current_line
                current_python_line += 1
            current_line += 1
        return -1

    def updateView(self):
        try:
            stack = self.gdb.get_current_stack()
            cpp_line = stack['line_no'] - 6 # First 5 lines of real C++ file are headers and stuff
            stack = self.pdb.get_current_stack()
            python_line = stack['line_no'] - 1 # Adjust to zero based
            self.current_lines = [self.get_line_from_cpp_line(cpp_line), self.get_line_from_python_line(python_line)]
        except:
            self.current_line = []
            self.button_step_over.setDisabled(True)

        hScrollBarPosition = self.textedit.horizontalScrollBar().value()
        vScrollBarPosition = self.textedit.verticalScrollBar().value()

        html = "<font face=\"" + self.fontFamily + "\" color=\"#000000\"><b>" + "<table><tr><td width=\"20\">";
        for line in range(len(self.mixedText.split("\n"))):
            if line in self.current_lines:
                html += "<img src=\"current_line.png\"/><br/>"
            else:
                html += "&nbsp;<br/>"
        html += "</td><td>" + self.render() + "</td></tr></table>" + "</b></font>"
        self.textedit.setHtml(html)

        self.textedit.horizontalScrollBar().setValue(hScrollBarPosition)
        self.textedit.verticalScrollBar().setValue(vScrollBarPosition)

        for watch in self.watch_values.keys():
            if watch[0] == 'C++':
                value = self.gdb.read_expr(watch[1])
                if value == None:
                    self.watch_values[watch].setText('')
                else:
                    self.watch_values[watch].setText(value)
            elif watch[0] == 'Python':
                value = self.pdb.read_expr(watch[1])
                self.watch_values[watch].setText(value)

    def render(self):
        x = ''
        for line in self.mixedText.split("\n"):
            line = cgi.escape(line).replace(' ', '&nbsp;')
            if len(line) > 1 and line[0] == '@':
                line = line[1:]
                x += "<font color=\"#606060\">" + line + "</font>" + "<br/>"
            else:
                x += line + "<br/>"
        return x

    def stepOver(self):
        # Get current lines
        stack = self.gdb.get_current_stack()
        cpp_line = stack['line_no'] - 6 # First 5 lines of real C++ file are headers and stuff
        stack = self.pdb.get_current_stack()
        python_line = stack['line_no'] - 1 # Adjust to zero based

        next_dict = {(5,4): (0,1), (5,5): (1,1),
                     (6,6): (0,1), (6,9): (0,1), (6,12): (1,1),
                     (7,13): (1,0), (8,13): (1,1),
                     (7,14): (1,0), (8,14): (1,1),
                     (7,15): (1,0), (8,15): (1,1),
                     (7,16): (1,0), (8,16): (2,1),
                     (6,19): (1,1)
                    }
        try:
            to_step = next_dict[tuple([python_line, cpp_line])]
        except:
            print 'Missing dictionary entry for line combination', tuple([python_line, cpp_line])
            return
        for x in range(to_step[0]):
            self.pdb.next()
        for x in range(to_step[1]):
            self.gdb.next()
        self.updateView()

    def restart(self):
        self.gdb.quit()
        self.pdb.quit()
        self.gdb = gdb.gdb()
        self.pdb = pdb.pdb()
        self.button_step_over.setEnabled(True)
        self.updateView()

app = QApplication(sys.argv)
form = Form()
form.show()
sys.exit(app.exec_())

