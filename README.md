# Ākau 3D — Desktop 3D bathymetry topography viewer

A fast,  3D viewer for seafloor/terrain data. Load XYZ/CSV, LAS/LAZ point clouds, or DEM GeoTIFFs, then grid, shade, and explore them in an interactive PyVista/Qt window.

 Stack: Python 3.10+, PySide6 (Qt), PyVista/VTX, PyVistaQt
  OS: Windows / macOS / Linux
- Use cases: quick‐look QC, presentations, stakeholder visuals, and lightweight analysis 

If you need ASCII-only, refer to the app as Akau 3D.



Features

- Readers
  - XYZ/CSV 
  - LAS/LAZ 
  - DEM GeoTIFF (via `rasterio`) with on-the-fly bilinear downsample
- Gridding & smoothing
  - Block-median prefilter (cell size control)
  - Regular grid build with interpolation (**linear / nearest / cubic**)
  - Automatic NaN handling; grid decimation if > 2 M cells for smooth interaction
- 3D rendering
  - Smooth shading by default; optional PBR 
  - Normal exaggeration control and silhouette 
  - Colormaps: viridis, turbo, terrain, cividis, plasma, magma, gray
  - Lighting controls sun azimuth/elevation/intensity, ambient
- Tools
  - Pick 2 points Profile distance vs. elevation/depth; uses SciPy if available
  - Toggle axes; Save screenshot
- UX
  - Max DEM size slider (keeps large rasters responsive)
  - Timing HUD (read / grid / build)


---

 Install

 Create a virtual environment
bash
 Windows 
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate



<img width="1490" height="978" alt="Screenshot 2025-10-01 165316" src="https://github.com/user-attachments/assets/b3480657-a5c8-4619-9504-6f1a2eb6aa81" />






