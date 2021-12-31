import visdom
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.vis.close()

        self.iters = {}
        self.lines = {}


    def display_current_results(self, iters, x, name='train_loss'):
        if name not in self.iters:
            self.inters[name] = []

        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        self.lines[name].append(x)


        self.vis.lines(x=np.array(self.iters[name]),
                       y=np.array(self.lines[name]),
                       win=name,
                       opts=dict(legend=[name], title=name))


    def display_roc(self, y_true, y_pred):
        fpr, tpr, ths = roc_curve(y_true, y_pred)
        self.vis.line(x=fpr,
                      y=tpr,
                      opts=dict(legend=['roc'],
                                title='roc'))

