#! /usr/bin/env python
#
#
#

from Tkinter import *
from matplotlib.backends import backend_tkagg
from matplotlib.backends.backend_tkagg import FigureManagerTkAgg

sub = Toplevel()
canvas = Canvas(sub, width=130, height=100)
c = canvas.create_rectangle
for i in range(5):
    rect = c(10+20*i, 20, 20*(i+1), 80)
canvas.pack()
app = FigureManagerTkAgg(canvas, 1, sub)
