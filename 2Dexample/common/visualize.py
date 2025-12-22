import numpy as np
import mouette as M
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.measure import marching_cubes

from .utils import forward_in_batches

def point_cloud_from_array(X, D=None):
    pc = M.mesh.from_arrays(X)
    if D is not None:
        pc.vertices.register_array_as_attribute("dist",D)
    return pc

def point_cloud_from_arrays(*args) -> M.mesh.PointCloud:
    clouds, labels = [], []
    for pts,label in args:
        clouds.append(point_cloud_from_array(pts))
        labels.append(np.full(pts.shape[0], fill_value=label))
    pc = M.mesh.merge(clouds)
    pc.vertices.register_array_as_attribute("label", np.concatenate(labels))
    return pc

def vector_field_from_array(pos, vec, scale=1.) -> M.mesh.PolyLine:
    pl = M.mesh.RawMeshData()
    if pos.shape[1]==2:
        pos = np.pad(pos, ((0,0),(0,1)))
        vec = np.pad(vec, ((0,0),(0,1)))
    n_pt = pos.shape[0]
    for i in range(n_pt):
        P1 = pos[i]
        P2 = pos[i] + scale * vec[i]
        pl.vertices += [P1, P2]
        pl.edges.append((2*i,2*i+1))
    return M.mesh.PolyLine(pl)


def render_sdf_2d(render_path, contour_path, gradient_path, model, domain: M.geometry.AABB,
                  device, res=1000, batch_size=1000, pi_ticks=True, pi_mode="0-2pi"):
    assert domain.dim == 2
    
    X = np.linspace(domain.mini[0], domain.maxi[0], res)
    resY = round(res * domain.span[1] / domain.span[0])
    Y = np.linspace(domain.mini[1], domain.maxi[1], resY)
    
    pts = np.hstack((np.meshgrid(X, Y))).swapaxes(0, 1).reshape(2, -1).T
    if gradient_path is not None:
        dist_values, grad_values = forward_in_batches(
            model, pts, device,
            compute_grad=True, batch_size=batch_size)
    else:
        dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
        
    img = np.concatenate(dist_values).reshape((res, resY)).T
    img = img[::1, :]
    
    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin > 0 or vmax < 0:
        vmin, vmax = -1, 1
        
    extent = [domain.mini[0], domain.maxi[0], domain.mini[1], domain.maxi[1]]
    
    def _apply_pi_ticks(ax):
        # 显示成 [-pi, pi] 风格的标签
        labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
        
        if pi_mode == "0-2pi":
            # 轴真实范围是 [0, 2pi]，但显示成 [-pi, pi]
            ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        elif pi_mode == "-pi-pi":
            # 轴真实范围就是 [-pi, pi]
            ticks = [-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi]
        else:
            return  # 未知模式就不动
        
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        
    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(img, cmap="seismic", norm=norm, origin='lower', extent=extent)
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        
        if pi_ticks:
            _apply_pi_ticks(ax)
            
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if contour_path is not None:
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm, origin='lower', extent=extent)
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        
        if pi_ticks:
            _apply_pi_ticks(ax)
            
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        
        XX, YY = np.meshgrid(X, Y)
        plt.contour(XX, YY, img, levels=16, colors='k', linestyles="solid", linewidths=0.3)
        plt.contour(XX, YY, img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        
    if gradient_path is not None:
        grad_norms = np.linalg.norm(grad_values, axis=1)
        grad_img = grad_norms.reshape((res, resY)).T
        grad_img = grad_img[::-1, :]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))
        
        plt.clf()
        pos = plt.imshow(grad_img, vmin=0.5, vmax=1.5, cmap="bwr", origin='lower', extent=extent)
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        
        if pi_ticks:
            _apply_pi_ticks(ax)
            
        XX, YY = np.meshgrid(X, Y)
        plt.contour(XX, YY, img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def render_sdf_quad(render_path, contour_path, gradient_path, model, P0, P1, P2, device, res=800, batch_size=1000, **kwargs):
    """
    Renders a 3D sdf along a specified (planar) quad in space.

    Args:
        render_path (str): Path to export a color render
        contour_path (str): Path to export the contour plot
        gradient_path (str): Path to export the gradient norm plot
        model : SDF neural model
        P0 (array): First point of the quad ( (0,0) in parameter space)
        P1 (array): Second point of the quad ( (1,0) in parameter space)
        P2 (array): Thirs point of the quad ( (0,1) in parameter space)
        device (str): cpu or cuda
        res (int, optional): Image resolution. Defaults to 800.
        batch_size (int, optional): Size of forward batches. Defaults to 1000.
    """
    
    dx = P1 - P0
    dy = P2 - P0
    X = np.linspace(0,1, res)
    resY = round(res * M.geometry.norm(dy)/M.geometry.norm(dx))
    Y = np.linspace(0,1, resY)
    extent = [0, 1, 0, 1]

    pts = []
    for ax in X:
        for ay in Y:
            p = P0 + ax*dx + ay*dy
            pts.append(p)
    pts = np.array(pts)
    
    if gradient_path is not None:
        dist_values,grad_values = forward_in_batches(
            model, pts, device, 
            compute_grad=True, batch_size=batch_size)
    else:
        dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)


    img = np.concatenate(dist_values).reshape((res,resY)).T
    img = img[::-1,:]

    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1
    print(vmin, vmax)

    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(
            img,
            cmap="seismic",
            norm=norm,
            origin='lower',
            extent=extent
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches='tight', pad_inches=0)

    if contour_path is not None:
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm, origin='lower', extent=extent)
        
        plt.xlabel("x")
        plt.ylabel("y")
        # cs = plt.contourf(X,-Y,img, levels=np.linspace(-0.1,0.1,11), cmap="seismic", extend="both")
        # cs.changed()
        plt.gca().set_aspect("equal", adjustable="box")
        XX, YY = np.meshgrid(X, Y)
        plt.contour(XX, YY, img, levels=kwargs.get("n_levels", 16), colors='k', linestyles="solid", linewidths=0.5)
        plt.contour(XX, YY, img, levels=[kwargs.get("zero_contour_offset", 0.)], colors='k', linestyles="solid", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches='tight', pad_inches=0, dpi=kwargs.get("dpi", 500))

    if gradient_path is not None:
        grad_norms = np.linalg.norm(grad_values,axis=1)
        grad_img = grad_norms.reshape((res,resY)).T
        grad_img = grad_img[::-1,:]
        print("GRAD NORM INTERVAL", (np.min(grad_img), np.max(grad_img)))

        plt.clf()
        pos = plt.imshow(grad_img, vmin=0.5, vmax=1.5, cmap="bwr", origin='lower', extent=extent)
        plt.gca().set_aspect("equal", adjustable="box")
        XX, YY = np.meshgrid(X, Y)
        plt.contour(XX, YY, img, levels=[0.], colors='k', linestyles="solid", linewidths=0.6)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches='tight', pad_inches=0)


def parameter_singular_values(model):
    layers = list(model.children())
    data= []
    for layer in layers:
        if hasattr(layer, "weight"):
            w = layer.weight
            u, s, v = torch.linalg.svd(w)
            # data.append(f"{layer}, {s}")
            data.append(f"{layer}, min={s.min()}, max={s.max()}")
    return data


def reconstruct_surface_marching_cubes(model, domain, device, iso=0, res=100, batch_size=5000):
    if isinstance(iso, (int,float)): iso = [iso]
    
    ### Feed grid to model
    L = [np.linspace(domain.mini[i], domain.maxi[i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
    dist_values = dist_values.reshape((res,res,res))

    ### Call marching cubes
    to_save = dict()
    for ioff,off in enumerate(iso):
        try:
            verts,faces,normals,values = marching_cubes(dist_values, level=off)
            values = values[:, np.newaxis]
            m = M.mesh.RawMeshData()
            m.vertices += list(verts)
            m.faces += list(faces)
            m = M.mesh.SurfaceMesh(m)
            normal_attr = m.vertices.create_attribute("normals", float, 3, dense=True)
            normal_attr._data = normals
            values_attr = m.vertices.create_attribute("values", float, 1, dense=True)
            values_attr._data = values
            to_save[(ioff, off)] = m
        except ValueError:
            continue
    
    ### Reproject meshes to correct coordinates
    for key,mesh in to_save.items():
        for v in mesh.id_vertices:
            pV = M.Vec(mesh.vertices[v])
            # 将 verts 映射到实际坐标
            verts = np.array(verts)  # shape (N,3)
            for i in range(3):
                verts[:, i] = np.interp(verts[:, i], np.arange(res), L[i])
                
            # 更新 Mouette 网格
            m.vertices = [M.Vec(*v) for v in verts]

        to_save[key] = mesh
    return to_save