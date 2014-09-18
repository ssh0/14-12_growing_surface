#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, August 2014.
#

from Tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time


class Eden:

    def __init__(self, L=32, T=1000, view=True, animation=True, surface=True):
        self.sub = None
        self.lattice = None
        self.time_delay = 0
        self.L = L  # lattice size
        self.T = T
        self.view = view
        self.animation = animation
        self.surface = surface

        if self.view:
            default_size = 630  # default size of canvas
            L = self.L
            Ly = int(self.T / L + 1.6 * self.T ** (1. / 3))
            self.rsize = int(default_size / (2 * max(L, Ly)))
            if self.rsize == 0:
                self.rsize = 1
            self.fig_size_x = 2 * self.rsize * L
            self.fig_size_y = 2 * self.rsize * Ly
            self.margin = 10
            sub = Toplevel()

            self.canvas = Canvas(sub, width=self.fig_size_x + 2 * self.margin,
                                 height=self.fig_size_y + 2 * self.margin)
            self.c = self.canvas.create_rectangle
            self.update = self.canvas.update
            if self.animation:
                self.c(self.margin - 1, self.margin,
                       self.fig_size_x + self.margin,
                       self.fig_size_y + self.margin,
                       outline='black', fill='white')

            self.canvas.pack()

    def grow_lattice(self):
        self.lattice = np.zeros([self.L, 3], dtype=int)
        self.lattice[:, 0] = 1
        self.lattice[:, 1] = -1
        if self.sub is None or not self.sub.winfo_exists():
            lattice = self.lattice
            L = self.L
            choice = random.choice
            ne = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            nnsite = set([(x, 1) for x in range(L)])
            if self.view and self.animation:
                site = [(x, 0) for x in range(L)]
                self.update_canvas(site)
                self.update_canvas(list(nnsite), color='cyan')
                self.update()
            t = []  # time
            S = []  # a number of growing sites
            N = []  # a number of occupied sites
            h = np.array([1] * self.L)
            omega = []
            hmax = 0
            _t = 0
            while _t < self.T:
                t.append(_t)
                S.append(len(nnsite))
                N.append(np.sum(lattice == 1))
                omega.append(np.std(h))
                nn = choice(list(nnsite))
                nnsite.remove(nn)
                lattice[nn] = 1
                i, j = nn
                if j + 1 > hmax:
                    hmax = j + 1
                    lattice = np.append(lattice, np.zeros([L, 1]), axis=1)
                newnn = set([((i + nx) % L, j + ny) for nx, ny in ne
                             if lattice[(i + nx) % L, j + ny] == 0])
                ss = newnn - nnsite
                nnsite = nnsite | newnn
                for m in list(ss):
                    lattice[m] = -1
                    if m[1] > h[m[0]]:
                            h[m[0]] = m[1]
                if self.view and self.animation:
                    self.update_canvas([nn])
                    self.update_canvas(list(ss), color='cyan')
                    self.update()
                _t += 1
            else:
                if self.view:
                    self.canvas.delete("all")
                    occupied = np.where(lattice == 1)
                    occupied = [(m, n)
                                for m, n in zip(occupied[0], occupied[1])]
                    neighber = np.where(lattice == -1)
                    neighber = [(m, n)
                                for m, n in zip(neighber[0], neighber[1])]
                    self.c(self.margin - 1, self.margin,
                           self.fig_size_x + self.margin,
                           self.fig_size_y + self.margin,
                           outline='black', fill='white')
                    self.update_canvas(occupied, color='black')
                    self.update_canvas(neighber, color='cyan')
                    if self.surface:
                        surface = [(x, h[x]) for x in range(L)]
                        self.update_canvas(surface, color='red')
                    self.update()

                    print "done: L = %d, T = %d" % (self.L, self.T)
            self.lattice = lattice

        return t, S, N, omega

    def update_canvas(self, site, color='black'):
        for m, n in site:
            self.c(2 * m * self.rsize + self.margin,
                   self.fig_size_y + self.margin -
                   2 * (n + 1) * self.rsize,
                   2 * (m + 1) * self.rsize + self.margin - 1,
                   self.fig_size_y + self.margin - 2 * n * self.rsize - 1,
                   outline=color, fill=color)
        if self.time_delay != 0:
            time.sleep(self.time_delay)


class TopWindow:

    def show_setting_window(self, title='', parameters=None, modes=[],
                            buttons=[]):
        self.root = Tk()
        self.root.title(title)

        frame1 = Frame(self.root, padx=5, pady=5)
        frame1.pack(side='top')
        self.entry = []
        for i, parameter in enumerate(parameters):
            label = Label(frame1, text=parameter.items()[0][0] + ' = ')
            label.grid(row=i, column=0, sticky=E)
            self.entry.append(Entry(frame1, width=10))
            self.entry[i].grid(row=i, column=1)
            self.entry[i].delete(0, END)
            self.entry[i].insert(0, parameter.items()[0][1])
        self.entry[0].focus_set()

        self.v = []
        for text, default in modes:
            self.v.append(BooleanVar())
            self.v[-1].set(default)
            self.b = Checkbutton(self.root, text=text, variable=self.v[-1])
            self.b.pack(anchor=W)

        for args in buttons:
            frame = Frame(self.root, padx=5, pady=5)
            frame.pack(side='left')
            for arg in args:
                b = Button(frame, text=arg[0], command=arg[1])
                b.pack(expand=YES, fill='x')

        f = Frame(self.root, padx=5, pady=5)
        f.pack(side='right')
        Button(f, text='quit', command=self.quit).pack(expand=YES, fill='x')

        self.root.mainloop()

    def quit(self):
        self.root.destroy()
        sys.exit()


class Main(object):

    def __init__(self):
        global top
        self.eden = None
        top = TopWindow()
        title = "Growing Surface"
        parameters = [{'L': 100}, {'T': 1000}, {'time delay': 0}]
        checkbuttons = [('animation', True), ('show surface', True)]
        buttons = [(('a: run', self.pushed), ('save', self.pr)),
                   (('b: beta', self.exp_b_beta),
                    ('b: alpha', self.exp_b_alpha),
                       ('fit', self.fitting)),
                   (('c: graph', self.exp_c),)]
        top.show_setting_window(title, parameters, checkbuttons, buttons)

    def pushed(self):
        L = int(top.entry[0].get())
        T = int(top.entry[1].get())
        self.eden = Eden(L, T)
        self.eden.animation = top.v[0].get()
        self.eden.surface = top.v[1].get()
        self.eden.time_delay = float(top.entry[2].get())
        self.eden.grow_lattice()

    def exp_b_beta(self):
        T = 10000  # 100000
        self.Llist = [32, 64, 128]
        self.t = [2 ** i for i in range(int(np.log2(T) + 1)) if 2 ** i <= T]
        self.exp_b(r'$t$', r'$\omega(t)$', 15, T, target='beta')

    def exp_b_alpha(self):
        T = 200000
        self.Llist = [2 ** i for i in range(1, 11)]
        self.t = [i for i in xrange(T)]
        self.exp_b(r'$L$', r'$\omega(L)$', 15, T, target='alpha')

    def exp_b(self, xlabel, ylabel, trials, T, target=''):
        if target == 'beta':
            self.target = 'beta'
        elif target == 'alpha':
            self.target = 'alpha'
        elif target == 'c':
            self.target = 'c'
            self.data = []
        else:
            return

        fig = plt.figure("Growing Surface")
        self.ax = fig.add_subplot(111)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel(xlabel, fontsize=16)
        self.ax.set_ylabel(ylabel, fontsize=16)
        self.ax.set_ymargin(0.05)
        self.omega = []
        for L in self.Llist:
            np_omega = np.array([])
            for trial in range(trials):
                eden = Eden(L, view=False)
                eden.T = T
                _t, S, N, omega = eden.grow_lattice()
                if self.target == 'alpha':
                    np_omega = np.append(np_omega, omega[-1])
                else:
                    omega = [omega[o] for o in self.t]
                    np_omega = np.append(np_omega, omega)
                    np_omega = np_omega.reshape(trial + 1, len(self.t))

            if self.target == 'alpha':
                self.omega.append(np.average(np_omega))
            else:
                self.omega = np.sum(np_omega, axis=0) / trials
                if self.target == 'c':
                    data = [(t / L ** 1.5, o / L ** (1. / 3))
                            for t, o in zip(self.t, self.omega)]
                    data.sort()
                    x, y = [], []
                    for d in data:
                        x.append(d[0])
                        y.append(d[1])
                    self.ax.plot(x, y, '-o', label='L = %d' % L)
                else:
                    self.ax.plot(self.t, self.omega, '-o', label='L = %d' % L)
        if self.target == 'alpha':
            self.ax.plot(self.Llist, self.omega, '-o')
        else:
            plt.legend(loc='best')

        fig.tight_layout()
        plt.show()

    def fitting(self):
        if self.target is None:
            return
        import scipy.optimize as optimize

        def fit_func(parameter0, x, omega):
            log = np.log
            c1 = parameter0[0]
            c2 = parameter0[1]
            residual = log(omega) - c1 - c2 * log(x)
            return residual

        def fitted(x, c1, expo):
            return np.exp(c1) * (x ** expo)

        cut_from = int(raw_input("from ? (index) >>> "))
        cut_to = int(raw_input("to ? (index) >>> "))
        if self.target == 'beta':
            cut_x = np.array(self.t[cut_from:cut_to])
        if self.target == 'alpha':
            cut_x = np.array(self.Llist[cut_from:cut_to])
        cut_omega = np.array(self.omega[cut_from:cut_to])
        parameter0 = [0.1, 0.5]
        result = optimize.leastsq(
            fit_func, parameter0, args=(cut_x, cut_omega))
        c1 = result[0][0]
        expo = result[0][1]

        if self.target == 'beta':
            label = r'fit func: $\beta$ = %f' % expo
        if self.target == 'alpha':
            label = r'fit func: $\alpha$ = %f' % expo

        self.ax.plot(cut_x, fitted(cut_x, c1, expo), lw=2, label=label)
        plt.legend(loc='best')
        plt.show()

    def exp_c(self):
        T = 20000  # 100000
        self.Llist = [2 ** i for i in range(5, 15)]
        self.t = [2 ** i for i in range(int(np.log2(T) + 1)) if 2 ** i <= T]
        self.exp_b(r'$t/L^{\alpha/\beta}$', r'$\omega(L,t)/L^{\alpha}$',
                   15, T, target='c')

    def pr(self):
        import tkFileDialog
        import os

        if self.eden is None:
            print "first, you should run 'run'."
            return

        fTyp = [('eps flle', '*.eps'), ('all files', '*')]
        filename = tkFileDialog.asksaveasfilename(filetypes=fTyp,
                                                  initialdir=os.getcwd(),
                                                  initialfile="figure_1.eps")
        if filename is None:
            return
        try:
            self.eden.canvas.postscript(file=filename)
        except TclError:
            print """
            TclError: Cannot save the figure.
            Canvas Window must be alive for save."""
            return 1
if __name__ == '__main__':

    app = Main()
