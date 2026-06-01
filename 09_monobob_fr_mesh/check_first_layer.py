import pyvista as pv, numpy as np
from scipy.spatial import cKDTree

r = pv.OpenFOAMReader("case.foam"); r.set_active_time_value(0.0)
m = r.read()
internal = m["internalMesh"].compute_cell_sizes(length=False, area=False, volume=True)
V, ctr = internal["Volume"], internal.cell_centers().points

bnd = m["boundary"]["bob"].compute_cell_sizes(area=True)   # in ra m.keys() nếu tên block khác
A, fctr = bnd["Area"], bnd.cell_centers().points

_, idx = cKDTree(ctr).query(fctr)
t = V[idx]/A
print(f"t  min={t.min():.4e}  mean={t.mean():.4e}  max={t.max():.4e}")
print(f"y+ centroid spacing  mean={0.5*t.mean():.4e}")
