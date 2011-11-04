#!/usr/bin/python
# -*- coding: utf-8 -*-
 
# Based on http://developer.qt.nokia.com/wiki/PySideTutorials_Simple_Dialog

import cgi
import sys
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
        self.button1 = QPushButton("Step over")
        self.button1.clicked.connect(self.stepOver)
        self.button2 = QPushButton("Close")
        self.button2.clicked.connect(exit)
        self.currentLine = 20

        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.textedit)
        self.vlayout.addWidget(self.button1)
        self.vlayout.addWidget(self.button2)

        self.setLayout(self.vlayout)
        self.mixedText = """\
# Example from DARPA 2011 May 18 slides
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
        self.updateView()

    def updateView(self):
        hScrollBarPosition = self.textedit.horizontalScrollBar().value()
        vScrollBarPosition = self.textedit.verticalScrollBar().value()
        self.textedit.setHtml("<font face=\"" + self.fontFamily + "\" color=\"#000000\"><b>" +
            "<table><tr><td width=\"20\">" + "&nbsp;<br/>" * (self.currentLine - 1) + "<img src=\"current_line.png\"/></td><td>" +
            self.render(self.mixedText) +
            "</td></tr></table>" + "</b></font>")
        self.textedit.horizontalScrollBar().setValue(hScrollBarPosition)
        self.textedit.verticalScrollBar().setValue(vScrollBarPosition)

    def render(self, mixedText):
        x = ''
        for line in mixedText.split("\n"):
            line = cgi.escape(line).replace(' ', '&nbsp;')
            if len(line) > 1 and line[0] == '@':
                line = line[1:]
                x += "<font color=\"#606060\">" + line + "</font>" + "<br/>"
            else:
                x += line + "<br/>"
        return x

    def stepOver(self):
        self.currentLine += 1
        self.updateView()

app = QApplication(sys.argv)
form = Form()
form.show()
sys.exit(app.exec_())

