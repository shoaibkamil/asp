#!/usr/bin/python
# -*- coding: utf-8 -*-
 
# Based on http://developer.qt.nokia.com/wiki/PySideTutorials_Simple_Dialog

import cgi
import sys
import gdb
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

        self.watch_values = dict()

        self.watch_layout = QGridLayout()
        self.watch_layout.addWidget(QLabel("x1"), 0, 0)
        self.watch_values['x1'] = QLineEdit()
        self.watch_layout.addWidget(self.watch_values['x1'], 0, 1)
        self.watch_layout.addWidget(QLabel("x2"), 1, 0)
        self.watch_values['x2'] = QLineEdit()
        self.watch_layout.addWidget(self.watch_values['x2'], 1, 1)
        self.watch_layout.addWidget(QLabel("x3"), 2, 0)
        self.watch_values['x3'] = QLineEdit()
        self.watch_layout.addWidget(self.watch_values['x3'], 2, 1)
        self.watch_layout.addWidget(QLabel("_my_out_grid[x3]"), 3, 0)
        self.watch_values['_my_out_grid[x3]'] = QLineEdit()
        self.watch_layout.addWidget(self.watch_values['_my_out_grid[x3]'], 3, 1)

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

    def updateView(self):
        try:
            stack = self.gdb.get_current_stack()
            cpp_line = stack['line_no'] - 6 # First 6 lines of real C++ file are headers and stuff
            self.current_line = self.get_line_from_cpp_line(cpp_line)
        except:
            self.current_line = -1

        hScrollBarPosition = self.textedit.horizontalScrollBar().value()
        vScrollBarPosition = self.textedit.verticalScrollBar().value()

        html = "<font face=\"" + self.fontFamily + "\" color=\"#000000\"><b>" + "<table><tr><td width=\"20\">";
        if self.current_line == -1:
            html += "&nbsp;"
            self.button_step_over.setDisabled(True)
        else:
            html += "&nbsp;<br/>" * self.current_line + "<img src=\"current_line.png\"/>"
        html += "</td><td>" + self.render() + "</td></tr></table>" + "</b></font>"
        self.textedit.setHtml(html)

        self.textedit.horizontalScrollBar().setValue(hScrollBarPosition)
        self.textedit.verticalScrollBar().setValue(vScrollBarPosition)

        for expr in self.watch_values.keys():
            value = self.gdb.read_expr(expr)
            if value == None:
                self.watch_values[expr].setText('')
            else:
                self.watch_values[expr].setText(value)

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
        self.gdb.next()
        self.updateView()

    def restart(self):
        self.gdb.quit()
        self.gdb = gdb.gdb()
        self.button_step_over.setEnabled(True)
        self.updateView()

app = QApplication(sys.argv)
form = Form()
form.show()
sys.exit(app.exec_())

