# Build from the repository root on Windows:
#   pyinstaller --clean --noconfirm packaging/PolypMonitor.spec
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# The build script changes into the repository root before invoking PyInstaller.
project = Path.cwd()
checkpoint = project / "model.pth"
if not checkpoint.is_file():
    raise SystemExit("model.pth is missing. Put the final checkpoint in the project root first.")

analysis = Analysis(
    [str(project / "app.py")],
    pathex=[str(project)],
    binaries=[],
    datas=[(str(checkpoint), ".")] + collect_data_files("torchvision"),
    hiddenimports=collect_submodules("torchvision"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "notebook", "IPython", "pandas", "scipy"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="PolypMonitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="PolypMonitor",
)
