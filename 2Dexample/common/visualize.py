import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .utils import forward_in_batches

def render_sdf_2d(render_path, model, domain,
                  device, res=1000, batch_size=1000):
    domain_min = np.asarray(domain[0])
    domain_max = np.asarray(domain[1])
    domain_span = domain_max - domain_min
    assert domain_min.shape == (2,) and domain_max.shape == (2,)
    
    X = np.linspace(domain_min[0], domain_max[0], res)
    resY = round(res * domain_span[1] / domain_span[0])
    Y = np.linspace(domain_min[1], domain_max[1], resY)
    
    pts = np.hstack((np.meshgrid(X, Y))).swapaxes(0, 1).reshape(2, -1).T
    dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
        
    img = np.concatenate(dist_values).reshape((res, resY)).T
    img = img[::1, :]
    
    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin > 0 or vmax < 0:
        vmin, vmax = -1, 1
        
    extent = [domain_min[0], domain_max[0], domain_min[1], domain_max[1]]
    
    def _apply_pi_ticks(ax):
        x_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        x_labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
        y_ticks = [np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        y_labels = [r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        
    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(img, cmap="seismic", norm=norm, origin='lower', extent=extent)
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        
        _apply_pi_ticks(ax)
            
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches='tight', pad_inches=0)
        plt.close()
