from cbx.plotting import PlotDynamicHistory
import matplotlib.pyplot as plt
import numpy as np

class PlotMirrorDynamicHistory:
    def __init__(self, dyn, **kwargs):
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        self.phx = PlotDynamicHistory(dyn, ax = ax[0], **kwargs)
        z = np.array(dyn.history['y'])
        ymin,ymax = [z.min(), z.max()]
        self.phy = PlotDynamicHistory(dyn, ax = ax[1], 
                                      particles = dyn.history['y'],
                                      **{k:v for k,v in kwargs.items() if not k in ['plot_drift']}, 
                                      objective_args={'x_min':ymin, 'x_max':ymax})
        
    def run_plots(self, freq=5, wait=0.1, save_args=None):
        for i in range(0, self.phx.max_it, int(freq)):
            for phz in [self.phx, self.phy]:
                phz.plot_at_ind(i)
                phz.decorate_at_ind(i)
            plt.pause(wait)