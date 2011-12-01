#!/usr/bin/python
# -*- coding: utf-8 -*-
 
# Based on http://developer.qt.nokia.com/wiki/PySideTutorials_Simple_Dialog

import cgi
import sys
import gdb
import pdb
import datetime
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtDeclarative import QDeclarativeView

def log(str):
    print datetime.datetime.now() - start_time, str

class WatchRemover(object):
    def __init__(self, form, row):
        self.form = form
        self.row = row

    def remove_watch(self):
        log('Removing watch ' + self.form.watches[self.row][0] + ", " + self.form.watches[self.row][1])
        self.form.watches.pop(self.row)
        self.form.update_watch_widgets()

class AddWatchForm(QDialog):
    def __init__(self, cpponly=False, parent=None):
        self.fontFamily = "Courier"
        super(AddWatchForm, self).__init__(parent)
        self.setWindowTitle("Add Watch")

        if not cpponly:
            self.layout_lang = QHBoxLayout()
            self.radiobutton_cpp = QRadioButton('C++')
            self.radiobutton_cpp.setChecked(True)
            self.radiobutton_python = QRadioButton('Python')
            self.layout_lang.addWidget(self.radiobutton_cpp)
            self.layout_lang.addWidget(self.radiobutton_python)

        self.layout_expr = QHBoxLayout()
        self.layout_expr.addWidget(QLabel('Expression to watch'))
        self.lineedit_expr = QLineEdit()
        self.layout_expr.addWidget(self.lineedit_expr)

        self.button_step_over = QPushButton("Add")
        self.button_step_over.clicked.connect(self.accept)
        self.button_close = QPushButton("Cancel")
        self.button_close.clicked.connect(self.reject)

        self.right_panel = QVBoxLayout()
        self.right_panel.addStretch()

        self.button_hlayout = QHBoxLayout()
        self.button_hlayout.addWidget(self.button_step_over)
        self.button_hlayout.addWidget(self.button_close)

        self.vlayout = QVBoxLayout()
        if not cpponly:
            self.vlayout.addLayout(self.layout_lang)
        self.vlayout.addLayout(self.layout_expr)
        self.vlayout.addLayout(self.button_hlayout)

        self.setLayout(self.vlayout)
        self.mixedText = mixedText

class Form(QDialog):
    def __init__(self, mixedText, python_file_gdb, cpp_file, cpp_start_line, python_file_pdb, python_start_line, next_dict, cpp_offset_lines=0, python_offset_lines=0, parent=None, watches=[], cpponly=False):
        self.python_file_gdb = python_file_gdb
        self.cpp_file = cpp_file
        self.cpp_start_line = cpp_start_line
        self.next_dict = next_dict
        self.python_file_pdb = python_file_pdb
        self.python_start_line = python_start_line
        self.cpp_offset_lines = cpp_offset_lines
        self.python_offset_lines = python_offset_lines
        self.watches = watches
        self.cpponly = cpponly

        self.fontFamily = "Courier"
        super(Form, self).__init__(parent)
        self.setWindowTitle("SEJITS Integrated Debugger")
        self.resize(1920, 1024)
        self.textedit = QTextEdit()
        self.textedit.setReadOnly(True)
        self.textedit.setLineWrapMode(QTextEdit.NoWrap)
        self.button_step_over = QPushButton("Step over")
        self.button_step_over.clicked.connect(self.stepOver)
        self.button_restart = QPushButton("Restart")
        self.button_restart.clicked.connect(self.restart)
        self.button_close = QPushButton("Close")
        self.button_close.clicked.connect(self.accept)

        self.lang_colors = dict()
        self.lang_colors['C++'] = '#ff0000'
        self.lang_colors['Python'] = '#0000ff'

        if self.cpponly:
            self.watches = [x for x in watches if x[0] == 'C++']
        self.watch_layout = QGridLayout()
        self.watch_values = dict()

        self.button_add_watch = QPushButton("Add watch")
        self.button_add_watch.clicked.connect(self.add_watch)

        self.right_panel = QVBoxLayout()
        self.right_panel.addLayout(self.watch_layout)
        self.right_panel.addWidget(self.button_add_watch)
        self.right_panel.addStretch()

        self.button_hlayout = QHBoxLayout()
        self.button_hlayout.addWidget(self.button_step_over)
        self.button_hlayout.addWidget(self.button_restart)
        self.button_hlayout.addWidget(self.button_close)

        self.edit_watch_hlayout = QHBoxLayout()
        self.edit_watch_hlayout.addWidget(self.textedit)
        self.edit_watch_hlayout.addLayout(self.right_panel)
        self.edit_watch_hlayout.setStretch(0, 3)
        self.edit_watch_hlayout.setStretch(1, 1)

        self.vlayout = QVBoxLayout()
        self.vlayout.addLayout(self.edit_watch_hlayout)
        self.vlayout.addLayout(self.button_hlayout)

        self.setLayout(self.vlayout)
        self.mixedText = mixedText
        self.gdb = gdb.gdb(self.python_file_gdb, self.cpp_file, self.cpp_start_line)
        self.pdb = pdb.pdb(self.python_file_pdb, self.python_start_line)
        self.updateView()
        self.update_watch_widgets()

    def emptyOutGrid(self, grid):
        for row in range(0, grid.rowCount()):
            for col in range(0, grid.columnCount()):
                item = grid.itemAtPosition(row, col)
                if item != None:
                    item.widget().hide()
                    grid.removeItem(item)

    def update_watch_widgets(self):
        self.emptyOutGrid(self.watch_layout)
        self.watch_values = dict()
        self.watch_removers = []
        row = 0
        for watch in self.watches:
            self.watch_values[watch] = QLineEdit()
            self.watch_layout.addWidget(QLabel('<font color="' + self.lang_colors[watch[0]] + '">' + watch[1] + '</font>'), row, 0)
            self.watch_layout.addWidget(self.watch_values[watch], row, 1)
            self.watch_layout.addWidget(QLabel('<font color="' + self.lang_colors[watch[0]] + '">' + watch[0] + '</font>'), row, 2)
            button_delete = QToolButton()
            button_delete.setIcon(QIcon('delete.png'))
            self.watch_removers.append(WatchRemover(self, row))
            button_delete.clicked.connect(self.watch_removers[-1].remove_watch)
            self.watch_layout.addWidget(button_delete, row, 3)
            row = row + 1
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
        if self.cpponly:
            return -1
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
            cpp_line = stack['line_no'] - 1 - self.cpp_offset_lines
            stack = self.pdb.get_current_stack()
            python_line = stack['line_no'] - 1 - self.python_offset_lines
            current_cpp_line = self.get_line_from_cpp_line(cpp_line)
            current_python_line = self.get_line_from_python_line(python_line)
        except Exception as e:
            print e
            current_cpp_line = -1
            current_python_line = -1
            self.button_step_over.setDisabled(True)

        hScrollBarPosition = self.textedit.horizontalScrollBar().value()
        vScrollBarPosition = self.textedit.verticalScrollBar().value()

        html = "<font face=\"" + self.fontFamily + "\"><b>" + "<table><tr><td width=\"20\">";
        for line in range(len(self.mixedText.split("\n"))):
            if line == current_cpp_line:
                html += "<img src=\"current_line_cpp.png\"/><br/>"
            elif line == current_python_line:
                html += "<img src=\"current_line_python.png\"/><br/>"
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
                    self.watch_values[watch].setText('<unavailable>')
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
                x += "<font color=\"" + self.lang_colors['C++'] + "\">" + line + "</font>" + "<br/>"
            else:
                if self.cpponly:
                    x += "<font color=\"" + self.lang_colors['C++'] + "\">// " + line + "</font>" + "<br/>"
                else:
                    x += "<font color=\"" + self.lang_colors['Python'] + "\">" + line + "</font>" + "<br/>"
        return x

    def stepOver(self):
        # Get current lines
        stack = self.gdb.get_current_stack()
        cpp_line = stack['line_no'] - 6 # First 5 lines of real C++ file are headers and stuff
        stack = self.pdb.get_current_stack()
        python_line = stack['line_no'] - 1 # Adjust to zero based

        try:
            if self.cpponly:
                to_step = [0,1]
            else:
                to_step = self.next_dict[tuple([python_line, cpp_line])]
        except:
            print 'Missing dictionary entry for line combination', tuple([python_line, cpp_line])
            return
        for x in range(to_step[0]):
            self.pdb.next()
        for x in range(to_step[1]):
            self.gdb.next()
        self.updateView()

        stack = self.gdb.get_current_stack()
        cpp_line_after = stack['line_no'] - 6 # First 5 lines of real C++ file are headers and stuff
        stack = self.pdb.get_current_stack()
        python_line_after = stack['line_no'] - 1 # Adjust to zero based
        log('Step over clicked, moved from lines (' + str(python_line) + ',' + str(cpp_line) + ') to lines (' + str(python_line_after) + ',' + str(cpp_line_after) + ')')

    def restart(self):
        log('Restart clicked')
        self.gdb.quit()
        self.pdb.quit()
        self.gdb = gdb.gdb(self.python_file_gdb, self.cpp_file, self.cpp_start_line)
        self.pdb = pdb.pdb(self.python_file_pdb, self.python_start_line)
        self.button_step_over.setEnabled(True)
        self.updateView()

    def add_watch(self):
        log('Add watch clicked')
        add_watch_form = AddWatchForm(cpponly=self.cpponly, parent=self)
        if add_watch_form.exec_() == QDialog.Accepted:
            lang = 'C++'
            if not self.cpponly and add_watch_form.radiobutton_python.isChecked():
                lang = 'Python'
            expr = add_watch_form.lineedit_expr.text()
            self.watches.append( (lang, expr) )
            self.update_watch_widgets()
            log('Add watch completed, added ' + lang + ', ' + expr)

start_time = datetime.datetime.now()
log('Starting')
app = QApplication(sys.argv)
example_num = int(sys.argv[1])
cpponly = (len(sys.argv) > 2 and sys.argv[2] == 'cpponly')
if example_num == 1:
    mixedText = """\
from stencil_kernel import *
import stencil_grid
import numpy

class ExampleKernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
@void kernel(PyObject *in_grid, PyObject *out_grid)
@{
@  #define _out_grid_array_macro(_d0,_d1) (_d1+(_d0 * 5))
@  #define _in_grid_array_macro(_d0,_d1) (_d1+(_d0 * 5))
@  npy_double *_my_out_grid = ((npy_double *)PyArray_DATA(out_grid));
@  npy_double *_my_in_grid = ((npy_double *)PyArray_DATA(in_grid));
        for x in out_grid.interior_points():
@  for (int x1 = 2; (x1 <= 4); x1 = (x1 + 1))
@  {
@    #pragma ivdep
@    for (int x2 = 2; (x2 <= 4); x2 = (x2 + 1))
@    {
@      int x3;
@      x3 = _out_grid_array_macro(x1, x2);
            for y in in_grid.neighbors(x, 1):
                out_grid[x] = out_grid[x] + in_grid[y]
@      _my_out_grid[x3] = (_my_out_grid[x3] + _my_in_grid[_in_grid_array_macro((x1 + 1), (x2 + 0))]);
@      _my_out_grid[x3] = (_my_out_grid[x3] + _my_in_grid[_in_grid_array_macro((x1 + -1), (x2 + 0))]);
@      _my_out_grid[x3] = (_my_out_grid[x3] + _my_in_grid[_in_grid_array_macro((x1 + 0), (x2 + 1))]);
@      _my_out_grid[x3] = (_my_out_grid[x3] + _my_in_grid[_in_grid_array_macro((x1 + 0), (x2 + -1))]);
@    }
@  }
@}

in_grid = StencilGrid([5,5])
for x in range(0,5):
    for y in range(0,5):
        in_grid.data[x,y] = x + y

out_grid = StencilGrid([5,5])
ExampleKernel().kernel(in_grid, out_grid)
"""
    next_dict = {(5,4): (0,1), (5,5): (1,1),
                 (6,6): (0,1), (6,9): (0,1), (6,12): (1,1),
                 (7,13): (1,0), (8,13): (1,1),
                 (7,14): (1,0), (8,14): (1,1),
                 (7,15): (1,0), (8,15): (1,1),
                 (7,16): (1,0), (8,16): (2,1),
                 (6,19): (1,1)
                }
    # watches = [('C++','x1'), ('C++', 'x2'), ('C++', 'x3'), ('C++', '_my_out_grid[x3]'),
    #            ('Python', 'x'), ('Python', 'y'), ('Python', 'out_grid[x]'), ('Python', 'in_grid[y]')]
    watches = []
    form = Form(mixedText, 'stencil_kernel_example.py', '/tmp/asp_cache/ca3a79b1ef34c14cdd7df371368ecb01/module.cpp', 7, 'stencil_kernel_example.2.py', ['break 17', 'continue', 's'], next_dict, cpp_offset_lines=5, watches=watches, cpponly=cpponly)
elif example_num == 2:
    mixedText = """\
from stencil_kernel import *
import sys
import numpy
import math

width = 50
height = 50
image_in = open('mallard_tiny.raw', 'rb')
stdev_d = 1
stdev_s = 70
radius = stdev_d * 3

class Kernel(StencilKernel):
   def kernel(self, in_img, filter_d, filter_s, out_img):
@  void kernel(PyObject *in_img, PyObject *filter_d, PyObject *filter_s, PyObject *out_grid)
@  {
@    #define _filter_s_array_macro(_d0) (_d0)
@    #define _in_img_array_macro(_d0,_d1) (_d1+(_d0 * 50))
@    #define _filter_d_array_macro(_d0) (_d0)
@    #define _out_grid_array_macro(_d0,_d1) (_d1+(_d0 * 50))
@    npy_double *_my_filter_s = ((npy_double *)PyArray_DATA(filter_s));
@    npy_double *_my_in_img = ((npy_double *)PyArray_DATA(in_img));
@    npy_double *_my_filter_d = ((npy_double *)PyArray_DATA(filter_d));
@    npy_double *_my_out_grid = ((npy_double *)PyArray_DATA(out_grid));
       for x in out_img.interior_points():
@    for (int x1 = 1; (x1 <= 48); x1 = (x1 + 1))
@    {
@      #pragma ivdep
@      for (int x2 = 1; (x2 <= 48); x2 = (x2 + 1))
@      {
@        int x3;
@        x3 = _out_grid_array_macro(x1, x2);
           for y in in_img.neighbors(x, 1):
               out_img[x] += in_img[y] * filter_d[int(distance(x, y))] * filter_s[abs(int(in_img[x]-in_img[y]))]
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 1), (x2 + 0))] * _my_filter_d[int(1)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 1), (x2 + 0))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + -1), (x2 + 0))] * _my_filter_d[int(1)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + -1), (x2 + 0))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 1))] * _my_filter_d[int(1)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 0), (x2 + 1))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + -1))] * _my_filter_d[int(1)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 0), (x2 + -1))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + -1), (x2 + -1))] * _my_filter_d[int(2)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + -1), (x2 + -1))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + -1), (x2 + 0))] * _my_filter_d[int(1)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + -1), (x2 + 0))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + -1), (x2 + 1))] * _my_filter_d[int(2)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + -1), (x2 + 1))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + -1))] * _my_filter_d[int(1)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 0), (x2 + -1))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] * _my_filter_d[int(0)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 1))] * _my_filter_d[int(1)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 0), (x2 + 1))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 1), (x2 + -1))] * _my_filter_d[int(2)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 1), (x2 + -1))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 1), (x2 + 0))] * _my_filter_d[int(1)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 1), (x2 + 0))])))]));
@        _my_out_grid[x3] = (_my_out_grid[x3] + ((_my_in_img[_in_img_array_macro((x1 + 1), (x2 + 1))] * _my_filter_d[int(2)]) * _my_filter_s[abs(int((_my_in_img[_in_img_array_macro((x1 + 0), (x2 + 0))] - _my_in_img[_in_img_array_macro((x1 + 1), (x2 + 1))])))]));
@      }
@    }
@  }

def gaussian(stdev, length):
    result = StencilGrid([length])
    scale = 1.0/(stdev*math.sqrt(2.0*math.pi))
    divisor = -1.0 / (2.0 * stdev * stdev)
    for x in xrange(length):
       result[x] = scale * math.exp(float(x) * float(x) * divisor)
    return result

pixels = map(ord, list(image_in.read(width * height))) # Read in grayscale values
intensity = float(sum(pixels))/len(pixels)

kernel = Kernel()
kernel.should_unroll = False
out_grid = StencilGrid([width,height])
out_grid.ghost_depth = radius
in_grid = StencilGrid([width,height])
in_grid.ghost_depth = radius
for x in range(-radius,radius+1):
    for y in range(-radius,radius+1):
        in_grid.neighbor_definition[1].append( (x,y) )

for x in range(0,width):
    for y in range(0,height):
        in_grid.data[(x, y)] = pixels[y * width + x]

kernel.kernel(in_grid, gaussian(stdev_d, radius*2), gaussian(stdev_s, 256), out_grid)

for x in range(0,width):
    for y in range(0,height):
        pixels[y * width + x] = out_grid.data[(x, y)]
out_intensity = float(sum(pixels))/len(pixels)
for i in range(0, len(pixels)):
    pixels[i] = min(255, max(0, int(pixels[i] * (intensity/out_intensity))))

image_out = open('out.raw', 'wb')
image_out.write(''.join(map(chr, pixels)))
"""
    next_dict = {(13, 6) : (0,1), (13, 7) : (0,1), (13, 8) : (0,1), (13, 9) : (1,1),
                 (14, 10): (0,1), (14, 13): (0,1), (14, 16): (1,1),
                 (15, 17): (1,0), (16, 17): (1,1), 
                 (15, 18): (1,0), (16, 18): (1,1), 
                 (15, 19): (1,0), (16, 19): (1,1), 
                 (15, 20): (1,0), (16, 20): (1,1), 
                 (15, 21): (1,0), (16, 21): (1,1), 
                 (15, 22): (1,0), (16, 22): (1,1), 
                 (15, 23): (1,0), (16, 23): (1,1), 
                 (15, 24): (1,0), (16, 24): (1,1), 
                 (15, 25): (1,0), (16, 25): (1,1), 
                 (15, 26): (1,0), (16, 26): (1,1), 
                 (15, 27): (1,0), (16, 27): (1,1), 
                 (15, 28): (1,0), (16, 28): (1,1), 
                 (15, 29): (1,0), (16, 29): (2,1), 
                }
    form = Form(mixedText, 'bilateral_filter.py', '/tmp/asp_cache/476da69a7cd754a699054871fbf8ae12/module.cpp', 12, 'bilateral_filter.2.py', ['break 44', 'continue', 's', 'r', 's', 'r', 's'], next_dict, cpp_offset_lines=5, cpponly=cpponly)
else:
    raise RuntimeError("Invalid example number, must be 1 or 2")
form.show()
result = app.exec_()
log('Exiting with result ' + str(result))
sys.exit(result)

