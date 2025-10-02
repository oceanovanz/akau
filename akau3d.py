#!/usr/bin/env python3
# Ākau 3D — Desktop-friendly 3D bathymetry/topography viewer
# (PySide6 + PyVista + PyVistaQt)
#
# This version is tuned for laptops/desktops (Windows/macOS/Linux):
#  - No Pi-specific GL overrides
#  - MSAA enabled (multi_samples=4)
#  - Smooth shading on by default; optional PBR
#  - Robust DEM downsample + NaN handling
#  - UI slider for Max DEM size; timing overlay
#
# If you need ASCII-only, you can set the window title to "Akau 3D" below.

import sys
import os
from pathlib import Path
import time
import numpy as np

# -------- Optional deps detection --------
try:
    from scipy.interpolate import griddata, RegularGridInterpolator
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import laspy  # noqa: F401
    HAVE_LASPY = True
except Exception:
    HAVE_LASPY = False

try:
    import rasterio  # noqa: F401
    HAVE_RASTERIO = True
except Exception:
    HAVE_RASTERIO = False

# -------- 3D / UI --------
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6 import QtWidgets
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QSlider,
    QGroupBox, QFormLayout, QSplitter, QMessageBox, QProgressDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat


# ---------------- Busy helper (simple loading/timing dialog) ----------------
class Busy:
    def __init__(self, parent: QMainWindow, message: str = "Working…"):
        self.parent = parent
        self.msg = message
        self.dlg: QProgressDialog | None = None
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.dlg = QProgressDialog(self.msg, None, 0, 0, self.parent)
        self.dlg.setWindowTitle("Please wait")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.dlg.setCancelButton(None)
        self.dlg.setMinimumDuration(0)
        self.dlg.setAutoClose(False)
        self.dlg.setAutoReset(False)
        self.dlg.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        return self

    def step(self, txt=None):
        if self.dlg:
            if txt:
                self.dlg.setLabelText(txt)
            QApplication.processEvents()

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.dlg:
                self.dlg.reset()
                self.dlg.close()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()


# ---------------- Data helpers ----------------

def read_xyz_like(fp: Path):
    """Reads plain XYZ/CSV with columns x y z (comma/space delimited)."""
    data = []
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                data.append((x, y, z))
            except ValueError:
                continue
    if not data:
        raise ValueError("No numeric XYZ rows found.")
    arr = np.asarray(data, dtype=np.float32)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def read_las_like(fp: Path):
    if not HAVE_LASPY:
        raise RuntimeError("laspy not installed; LAS/LAZ not supported.")
    import laspy
    with laspy.open(str(fp)) as l:
        pts = l.read()
    return pts.x.astype(np.float32), pts.y.astype(np.float32), pts.z.astype(np.float32)


def read_geotiff_dem(fp: Path, max_dim: int = 1500):
    """Read DEM GeoTIFF with on-the-fly downsample (bilinear). Returns float32 arrays."""
    if not HAVE_RASTERIO:
        raise RuntimeError("rasterio not installed; GeoTIFF not supported.")
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling

    with rasterio.open(fp) as src:
        band = 1
        nodata = src.nodata

        # Aspect-preserving target size
        scale = min(max_dim / src.width, max_dim / src.height, 1.0)
        out_w = max(1, int(src.width * scale))
        out_h = max(1, int(src.height * scale))

        arr = src.read(
            band,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        # Replace nodata with NaN then fill with median (avoids VTK crashes)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        if np.isnan(arr).any():
            med = np.nanmedian(arr)
            if not np.isfinite(med):
                med = 0.0
            arr = np.nan_to_num(arr, nan=float(med))

        # Transform for downsampled grid
        scale_x = src.width / out_w
        scale_y = src.height / out_h
        transform = src.transform * Affine.scale(scale_x, scale_y)

        xs = transform.c + np.arange(out_w, dtype=np.float32) * transform.a
        ys = transform.f + np.arange(out_h, dtype=np.float32) * transform.e
        XI, YI = np.meshgrid(xs, ys)
        return XI.astype(np.float32), YI.astype(np.float32), arr


def blockmedian_xyz(x, y, z, cell):
    xi = np.floor((x - x.min()) / cell).astype(np.int32)
    yi = np.floor((y - y.min()) / cell).astype(np.int32)
    key = xi.astype(np.int64) << 32 | yi.astype(np.int64)
    idx = np.argsort(key)
    key_sorted = key[idx]
    x_s, y_s, z_s = x[idx], y[idx], z[idx]
    uniq, start = np.unique(key_sorted, return_index=True)
    end = np.r_[start[1:], key_sorted.size]
    xm, ym, zm = [], [], []
    for s, e in zip(start, end):
        xm.append(np.median(x_s[s:e]))
        ym.append(np.median(y_s[s:e]))
        zm.append(np.median(z_s[s:e]))
    return np.asarray(xm, np.float32), np.asarray(ym, np.float32), np.asarray(zm, np.float32)


def grid_xyz(x, y, z, nx=1000, ny=1000, method='linear'):
    xi = np.linspace(x.min(), x.max(), int(nx), dtype=np.float32)
    yi = np.linspace(y.min(), y.max(), int(ny), dtype=np.float32)
    XI, YI = np.meshgrid(xi, yi)
    if HAVE_SCIPY:
        ZI = griddata((x, y), z, (XI, YI), method=method)
        if np.isnan(ZI).any():
            ZN = griddata((x, y), z, (XI, YI), method='nearest')
            ZI = np.where(np.isnan(ZI), ZN, ZI)
        return XI.astype(np.float32), YI.astype(np.float32), ZI.astype(np.float32)
    # Fallback triangulation
    cloud = pv.PolyData(np.c_[x, y, z])
    surf = cloud.delaunay_2d(alpha=0.0)
    probe = pv.StructuredGrid(XI, YI, np.zeros_like(XI))
    sampled = probe.sample(surf)
    ZI = sampled.points[:, 2].reshape(XI.shape)
    return XI.astype(np.float32), YI.astype(np.float32), ZI.astype(np.float32)


def make_structured_grid(XI, YI, ZI, invert=False):
    Z = (-ZI if invert else ZI).astype(np.float32, copy=False)
    return pv.StructuredGrid(XI.astype(np.float32), YI.astype(np.float32), Z)


# ---------------- Main Window ----------------
class MBViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        # Use macron form in UI; switch to "Akau 3D" if you need ASCII-only
        self.setWindowTitle("Ākau 3D")
        self.resize(1500, 950)

        self.current_grid = None
        self.current_arrays = None   # (XI, YI, ZI)
        self.interpolator = None
        self.pick_points = []
        self.last_dir = str(Path.cwd())

        # Splitter layout
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left: 3D view (enable MSAA)
        self.plotter = QtInteractor(self, auto_update=True, multi_samples=4)
        self.plotter.camera_position = 'iso'
        self.plotter.set_background("white", top="lightsteelblue")
        splitter.addWidget(self.plotter.interactor)
        splitter.setStretchFactor(0, 1)

        # Right: controls
        controls = self._build_controls()
        cw = QWidget(); cw.setLayout(controls); cw.setMinimumWidth(380)
        splitter.addWidget(cw)

        # Lighting
        self._setup_lighting()

        # HUD text
        self.hud_text = self.plotter.add_text("Load data to begin", font_size=12, color="black")

        # Timing last-run dict
        self.last_timing = {}

    # ---------- UI ----------
    def _build_controls(self):
        layout = QVBoxLayout()

        # Data group
        file_group = QGroupBox("Data")
        file_form = QVBoxLayout()

        b1 = QPushButton("Open XYZ / CSV"); b1.clicked.connect(self.open_xyz); file_form.addWidget(b1)
        b2 = QPushButton("Open LAS / LAZ"); b2.clicked.connect(self.open_las); file_form.addWidget(b2)
        b3 = QPushButton("Open DEM (GeoTIFF)"); b3.clicked.connect(self.open_tif); file_form.addWidget(b3)

        file_group.setLayout(file_form)
        layout.addWidget(file_group)

        # Gridding
        grid_group = QGroupBox("Gridding / DEM")
        grid_form = QFormLayout()

        self.spin_res = QDoubleSpinBox(); self.spin_res.setDecimals(2); self.spin_res.setRange(0.1, 2000.0); self.spin_res.setValue(2.0); self.spin_res.setSuffix(" m cell")
        grid_form.addRow("Blockmedian cell:", self.spin_res)

        self.spin_nx = QSpinBox(); self.spin_nx.setRange(50, 6000); self.spin_nx.setValue(1200)
        self.spin_ny = QSpinBox(); self.spin_ny.setRange(50, 6000); self.spin_ny.setValue(1200)
        h = QHBoxLayout(); h.addWidget(QLabel("NX")); h.addWidget(self.spin_nx); h.addSpacing(12); h.addWidget(QLabel("NY")); h.addWidget(self.spin_ny)
        grid_form.addRow("Output grid:", h)

        self.combo_method = QComboBox(); self.combo_method.addItems(["linear", "nearest", "cubic"]) ; grid_form.addRow("Interpolation:", self.combo_method)

        self.chk_invert = QCheckBox("Invert Z (depth→up)"); grid_form.addRow(self.chk_invert)

        # New: Max DEM size slider (controls GeoTIFF downsample)
        self.dem_max = QSlider(Qt.Horizontal); self.dem_max.setRange(400, 3000); self.dem_max.setValue(1500)
        self.dem_max.setSingleStep(50)
        grid_form.addRow("Max DEM size (px):", self.dem_max)

        btn_regrid = QPushButton("Rebuild Surface"); btn_regrid.clicked.connect(self.regrid_surface)
        grid_form.addRow(btn_regrid)

        grid_group.setLayout(grid_form)
        layout.addWidget(grid_group)

        # Appearance
        app_group = QGroupBox("Appearance")
        app_form = QFormLayout()

        self.combo_cmap = QComboBox(); self.combo_cmap.addItems(["viridis", "turbo", "terrain", "cividis", "plasma", "magma", "gray"])
        self.combo_cmap.currentTextChanged.connect(self.update_colormap)
        app_form.addRow("Colormap:", self.combo_cmap)

        self.chk_pbr = QCheckBox("Enable PBR material"); self.chk_pbr.setChecked(True); self.chk_pbr.stateChanged.connect(self.update_pbr)
        app_form.addRow(self.chk_pbr)

        self.roughness = QDoubleSpinBox(); self.roughness.setRange(0.0, 1.0); self.roughness.setSingleStep(0.05); self.roughness.setValue(0.75); self.roughness.valueChanged.connect(self.update_pbr)
        app_form.addRow("Roughness:", self.roughness)

        self.metallic = QDoubleSpinBox(); self.metallic.setRange(0.0, 1.0); self.metallic.setSingleStep(0.05); self.metallic.setValue(0.1); self.metallic.valueChanged.connect(self.update_pbr)
        app_form.addRow("Metallic:", self.metallic)

        self.normal_scale = QDoubleSpinBox(); self.normal_scale.setRange(0.0, 5.0); self.normal_scale.setSingleStep(0.1); self.normal_scale.setValue(0.0); self.normal_scale.valueChanged.connect(self.update_normals)
        app_form.addRow("Normal exaggeration:", self.normal_scale)

        self.chk_edges = QCheckBox("Edge enhance (silhouette)"); self.chk_edges.stateChanged.connect(self.update_edges)
        app_form.addRow(self.chk_edges)

        app_group.setLayout(app_form)
        layout.addWidget(app_group)

        # Lighting
        light_group = QGroupBox("Lighting")
        lform = QFormLayout()
        self.sun_elev = QSlider(Qt.Horizontal); self.sun_elev.setRange(0, 90); self.sun_elev.setValue(45); self.sun_elev.valueChanged.connect(self.update_light)
        lform.addRow("Sun elevation:", self.sun_elev)
        self.sun_az = QSlider(Qt.Horizontal); self.sun_az.setRange(0, 360); self.sun_az.setValue(135); self.sun_az.valueChanged.connect(self.update_light)
        lform.addRow("Sun azimuth:", self.sun_az)
        self.sun_int = QSlider(Qt.Horizontal); self.sun_int.setRange(0, 200); self.sun_int.setValue(120); self.sun_int.valueChanged.connect(self.update_light)
        lform.addRow("Sun intensity:", self.sun_int)
        self.amb_int = QSlider(Qt.Horizontal); self.amb_int.setRange(0, 200); self.amb_int.setValue(40); self.amb_int.valueChanged.connect(self.update_light)
        lform.addRow("Ambient intensity:", self.amb_int)
        light_group.setLayout(lform)
        layout.addWidget(light_group)

        # Tools
        tool_group = QGroupBox("Tools")
        tl = QVBoxLayout()
        btn_profile = QPushButton("Pick 2 points → Profile"); btn_profile.clicked.connect(self.start_pick_profile); tl.addWidget(btn_profile)
        self.chk_show_axes = QCheckBox("Show axes"); self.chk_show_axes.stateChanged.connect(lambda s: self.plotter.show_axes() if s else self.plotter.hide_axes()); tl.addWidget(self.chk_show_axes)
        btn_shot = QPushButton("Save Screenshot…"); btn_shot.clicked.connect(self.save_screenshot); tl.addWidget(btn_shot)
        tool_group.setLayout(tl)
        layout.addWidget(tool_group)

        layout.addStretch(1)
        return layout

    # ---------- File open ----------
    def open_xyz(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Open XYZ/CSV", self.last_dir, "XYZ/CSV (*.xyz *.txt *.csv);;All files (*.*)")
        if not fp:
            return
        self.last_dir = str(Path(fp).parent)
        t0 = time.perf_counter()
        try:
            with Busy(self, "Loading XYZ/CSV…"):
                x, y, z = read_xyz_like(Path(fp))
                self._ingest_points(x, y, z, src_name=Path(fp).name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load XYZ/CSV:\n{e}")
        finally:
            self._show_timing(read=time.perf_counter() - t0)

    def open_las(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Open LAS / LAZ", self.last_dir, "LAS/LAZ (*.las *.laz)")
        if not fp:
            return
        self.last_dir = str(Path(fp).parent)
        t0 = time.perf_counter()
        try:
            with Busy(self, "Loading LAS/LAZ…"):
                x, y, z = read_las_like(Path(fp))
                self._ingest_points(x, y, z, src_name=Path(fp).name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load LAS/LAZ:\n{e}")
        finally:
            self._show_timing(read=time.perf_counter() - t0)

    def open_tif(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Open DEM (GeoTIFF)", self.last_dir, "GeoTIFF (*.tif *.tiff)")
        if not fp:
            return
        self.last_dir = str(Path(fp).parent)
        t0 = time.perf_counter()
        try:
            with Busy(self, "Reading DEM…"):
                XI, YI, ZI = read_geotiff_dem(Path(fp), max_dim=int(self.dem_max.value()))
                self.current_arrays = (XI, YI, ZI)
            with Busy(self, "Building 3D surface…"):
                self._build_surface(update_camera=True, label=Path(fp).name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DEM:\n{e}")
        finally:
            self._show_timing(read=time.perf_counter() - t0)

    # ---------- Ingest & Grid ----------
    def _ingest_points(self, x, y, z, src_name="points"):
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[m], y[m], z[m]
        if x.size < 100:
            QMessageBox.warning(self, "Too few points", "Less than 100 valid points after cleaning.")
            return

        t_grid0 = time.perf_counter()
        # Prefilter with blockmedian
        cell = max(self.spin_res.value(), 0.1)
        with Busy(self, "Prefiltering points…"):
            xbm, ybm, zbm = blockmedian_xyz(x, y, z, cell)

        # Grid to regular mesh
        with Busy(self, "Gridding points…"):
            XI, YI, ZI = grid_xyz(
                xbm, ybm, zbm,
                nx=self.spin_nx.value(),
                ny=self.spin_ny.value(),
                method=self.combo_method.currentText()
            )
        self.current_arrays = (XI, YI, ZI)

        with Busy(self, "Building 3D surface…"):
            self._build_surface(update_camera=True, label=src_name)
        self._show_timing(grid=time.perf_counter() - t_grid0)

    def regrid_surface(self):
        if self.current_arrays is None:
            QMessageBox.information(self, "No data", "Load data first.")
            return
        t0 = time.perf_counter()
        with Busy(self, "Rebuilding 3D surface…"):
            self._build_surface(update_camera=False)
        self._show_timing(build=time.perf_counter() - t0)

    def _build_surface(self, update_camera=True, label=""):
        # Re-entrancy guard
        if getattr(self, "_building", False):
            return
        self._building = True
        t0 = time.perf_counter()

        XI, YI, ZI = self.current_arrays
        invert = self.chk_invert.isChecked()

        # Cap total points to ~2M for desktop smoothness; stride equally in x/y
        max_pts = 2_000_000
        npts = int(XI.size)
        if npts > max_pts:
            stride = int(np.ceil(np.sqrt(npts / max_pts)))
            XI = XI[::stride, ::stride]
            YI = YI[::stride, ::stride]
            ZI = ZI[::stride, ::stride]
            print(f"Decimated grid by stride={stride} -> {XI.shape}")

        # Ensure float32 and NaN-free
        XI = XI.astype(np.float32, copy=False)
        YI = YI.astype(np.float32, copy=False)
        ZI = ZI.astype(np.float32, copy=False)
        if np.isnan(ZI).any():
            med = np.nanmedian(ZI)
            if not np.isfinite(med):
                med = 0.0
            ZI = np.nan_to_num(ZI, nan=float(med))

        Z = -ZI if invert else ZI
        mesh = pv.StructuredGrid(XI, YI, Z)

        # Optionally compute normals for nicer shading
        try:
            mesh = mesh.compute_normals(consistent=True, auto_orient_normals=True, splitting=True, feature_angle=45.0)
        except Exception:
            pass

        # Render
        self.plotter.clear()
        self.current_grid = mesh

        elev = (-ZI if invert else ZI).astype(np.float32, copy=False)
        mesh["elev"] = elev.ravel(order="F")

        actor = self.plotter.add_mesh(
            mesh,
            scalars="elev",
            cmap=self.combo_cmap.currentText(),
            smooth_shading=True,
            show_edges=False,
        )
        try:
            if self.chk_pbr.isChecked():
                actor.prop.SetInterpolationToPBR()
                actor.prop.SetRoughness(self.roughness.value())
                actor.prop.SetMetallic(self.metallic.value())
            else:
                actor.prop.SetInterpolationToPhong()
        except Exception:
            pass

        self.update_edges()
        self.update_light()
        self.plotter.add_axes()
        self.chk_show_axes.setChecked(True)

        # Ground plane (subtle)
        try:
            b = mesh.bounds
            size = max(b[1]-b[0], b[3]-b[2]) * 1.2
            c = mesh.center
            plane = pv.Plane(center=(c[0], c[1], b[4]), direction=(0,0,1), i_size=size, j_size=size)
            self.plotter.add_mesh(plane, color="white", opacity=0.05)
        except Exception:
            pass

        self.hud_text = self.plotter.add_text(f"{label} — {XI.shape[1]}×{XI.shape[0]} grid", font_size=10, color="black")
        if update_camera:
            self.plotter.camera_position = 'iso'
            self.plotter.reset_camera()
        self.plotter.render()

        # Profile interpolator
        try:
            if HAVE_SCIPY:
                yi = YI[:, 0]
                xi = XI[0, :]
                self.interpolator = RegularGridInterpolator((yi, xi), ZI)
            else:
                self.interpolator = None
        except Exception:
            self.interpolator = None

        # record timing
        self._show_timing(build=time.perf_counter() - t0)
        self._building = False

    # ---------- Appearance ----------
    def update_colormap(self, *_):
        if self.current_grid is None:
            return
        self.plotter.update_scalar_bar_range()
        self.plotter.render()

    def update_pbr(self, *_):
        if self.current_grid is None:
            return
        for a in self.plotter.renderer.actors.values():
            p = a.GetProperty()
            try:
                if self.chk_pbr.isChecked():
                    p.SetInterpolationToPBR(); p.SetRoughness(self.roughness.value()); p.SetMetallic(self.metallic.value())
                else:
                    p.SetInterpolationToPhong()
            except Exception:
                pass
        self.plotter.render()

    def update_normals(self, *_):
        if self.current_grid is None:
            return
        scale = self.normal_scale.value()
        if scale <= 0:
            with Busy(self, "Resetting surface…"):
                self._build_surface(update_camera=False)
            return
        with Busy(self, "Applying normal exaggeration…"):
            warped = self.current_grid.warp_by_scalar(factor=scale*0.02, scalars=None)
            self.plotter.clear()
            actor = self.plotter.add_mesh(warped, scalars="elev", cmap=self.combo_cmap.currentText(), smooth_shading=True)
            try:
                if self.chk_pbr.isChecked():
                    actor.prop.SetInterpolationToPBR(); actor.prop.SetRoughness(self.roughness.value()); actor.prop.SetMetallic(self.metallic.value())
                else:
                    actor.prop.SetInterpolationToPhong()
            except Exception:
                pass
            self.update_edges(); self.update_light(); self.plotter.add_axes(); self.plotter.render()

    def update_edges(self, *_):
        if self.current_grid is None:
            return
        # Remove previous edge actor if present
        if hasattr(self, "_edge_actor") and self._edge_actor is not None:
            try:
                self.plotter.remove_actor(self._edge_actor)
            except Exception:
                pass
            self._edge_actor = None
        if self.chk_edges.isChecked():
            try:
                edges = self.current_grid.extract_feature_edges(boundary_edges=False, feature_edges=True, non_manifold_edges=False, manifold_edges=False, feature_angle=45)
                self._edge_actor = self.plotter.add_mesh(edges, color="black", line_width=1, opacity=0.15)
            except Exception:
                self._edge_actor = None
        self.plotter.render()

    # ---------- Lighting ----------
    def _setup_lighting(self):
        self.sun = pv.Light(light_type='scenelight')
        self.sun.set_direction_angle(45, 135)
        self.sun.intensity = 1.2
        self.plotter.add_light(self.sun)
        self.amb = pv.Light(light_type='headlight')
        self.amb.intensity = 0.4
        self.plotter.add_light(self.amb)

    def update_light(self, *_):
        self.sun.set_direction_angle(self.sun_elev.value(), self.sun_az.value())
        self.sun.intensity = self.sun_int.value() / 100.0
        self.amb.intensity = self.amb_int.value() / 100.0
        self.plotter.render()

    # ---------- Tools ----------
    def start_pick_profile(self):
        if self.current_grid is None:
            QMessageBox.information(self, "No surface", "Load and build a surface first.")
            return
        if not HAVE_SCIPY:
            QMessageBox.information(self, "Profiles unavailable", "Install SciPy to enable the profile tool.")
            return
        if self.interpolator is None:
            QMessageBox.information(self, "Profiles unavailable", "Interpolator not ready; try rebuilding the surface.")
            return
        self.pick_points = []
        try:
            self.hud_text.SetText(0, "Click two points on the surface to create a profile…")
        except Exception:
            pass
        self.plotter.enable_point_picking(callback=self._on_pick, use_mesh=True, show_message=False, show_point=False, left_clicking=True)

    def _on_pick(self, point):
        self.pick_points.append(point)
        if len(self.pick_points) == 2:
            self.plotter.disable_picking()
            self._do_profile(self.pick_points[0], self.pick_points[1])
            self.pick_points = []
            try:
                self.hud_text.SetText(0, "Profile plotted. You can pick again.")
            except Exception:
                pass

    def _do_profile(self, p0, p1, n=400):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            QMessageBox.critical(self, "Matplotlib error", "Unable to import matplotlib:\n" + str(e))
            return
        XI, YI, ZI = self.current_arrays
        p0xy = np.array([p0[0], p0[1]]); p1xy = np.array([p1[0], p1[1]])
        ts = np.linspace(0, 1, n)
        px = p0xy[0] + (p1xy[0]-p0xy[0]) * ts
        py = p0xy[1] + (p1xy[1]-p0xy[1]) * ts
        samples = self.interpolator(np.c_[py, px])  # (Y,X) order
        dist = np.hypot(px - px[0], py - py[0])
        plt.figure(figsize=(8, 4)); plt.plot(dist, samples, linewidth=2)
        plt.xlabel("Distance (m)"); plt.ylabel("Depth" if not self.chk_invert.isChecked() else "Elevation"); plt.title("Profile"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
        line = pv.Line((p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2]), resolution=1)
        self.plotter.add_mesh(line, color="black", line_width=3, name="profile_line"); self.plotter.render()

    def save_screenshot(self):
        if self.current_grid is None:
            QMessageBox.information(self, "No surface", "Load and build a surface first.")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", self.last_dir, "PNG (*.png)")
        if not fp:
            return
        try:
            self.plotter.screenshot(fp, window_size=None, transparent_background=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save screenshot:\n{e}")

    # ---------- Timing HUD ----------
    def _show_timing(self, read=None, grid=None, build=None):
        if read is not None:
            self.last_timing['read'] = read
        if grid is not None:
            self.last_timing['grid'] = grid
        if build is not None:
            self.last_timing['build'] = build
        if not self.last_timing:
            return
        parts = []
        if 'read' in self.last_timing:
            parts.append(f"Read: {self.last_timing['read']*1000:.0f} ms")
        if 'grid' in self.last_timing:
            parts.append(f"Grid: {self.last_timing['grid']*1000:.0f} ms")
        if 'build' in self.last_timing:
            parts.append(f"Build: {self.last_timing['build']*1000:.0f} ms")
        txt = " · ".join(parts)
        try:
            self.hud_text.SetText(0, txt)
        except Exception:
            # If the text actor changed, re-add
            self.hud_text = self.plotter.add_text(txt, font_size=10, color="black")
        self.plotter.render()


# ---------------- Entrypoint ----------------
def main():
    pv.set_plot_theme("document")
    # Request some samples for MSAA
    fmt = QSurfaceFormat(); fmt.setSamples(4); QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    win = MBViewer(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
