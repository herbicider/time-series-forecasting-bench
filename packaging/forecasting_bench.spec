# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Time Series Forecasting Bench.

One spec covers both platforms and both editions. Choose with env vars:

    FB_EDITION=standard | ai      (default: standard)

  standard  No torch. Roughly 250 MB on Windows, 340 MB on macOS. Ships seven
            working forecasters and never touches the network.

  ai        Adds torch + timesfm + chronos (~1 GB). Model weights are NOT
            bundled; they download on first use into the per-user cache
            managed by core/models/manager.py.

Both editions build a *portable folder*, not an installer: the user unzips it
and double-clicks the executable. No admin rights, no setup wizard, nothing
written to Program Files or the registry.

Why not a single-file .exe? PyInstaller's one-file mode unpacks the entire
bundle to a temp directory on *every* launch. At this size that is a 10-20 s
wait each time the user opens the app, which reads as broken. The bulk is
unavoidable: statsforecast pulls in pyarrow (~97 MB) and llvmlite (~87 MB),
and removing either silently drops two of the seven forecasters.

Build:
    pyinstaller packaging/forecasting_bench.spec --noconfirm
"""

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# PyInstaller resolves relative paths against the spec file's own directory,
# not the working directory, so anchor everything to the repo root explicitly.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

EDITION = os.environ.get("FB_EDITION", "standard").lower()
IS_AI = EDITION == "ai"
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

APP_NAME = "ForecastingBench"
APP_DISPLAY_NAME = "Time Series Forecasting Bench"
VERSION = os.environ.get("FB_VERSION", "1.1.0")

# --------------------------------------------------------------------------
# Data files
# --------------------------------------------------------------------------

datas = [
    (os.path.join(ROOT, "ui"), "ui"),   # includes ui/vendor: echarts + fonts, required offline
    (os.path.join(ROOT, "samples"), "samples"),
]

for pkg in ("statsforecast", "utilsforecast", "coreforecast"):
    datas += collect_data_files(pkg, include_py_files=True)

# --------------------------------------------------------------------------
# Hidden imports
# --------------------------------------------------------------------------

hiddenimports = [
    # uvicorn resolves these by string at runtime
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # statsforecast model classes are looked up dynamically
    "statsforecast.models",
    # reportlab submodules used by the PDF export
    "reportlab.platypus",
    "reportlab.lib.pagesizes",
    "reportlab.lib.colors",
    "reportlab.lib.styles",
    # pywebview picks its GUI backend at runtime; without these the packaged
    # app starts and then fails to open a window at all.
    "webview.platforms",
]

if IS_WINDOWS:
    hiddenimports += ["webview.platforms.edgechromium", "webview.platforms.winforms",
                      "clr", "System"]
if IS_MAC:
    hiddenimports += ["webview.platforms.cocoa", "objc", "Foundation", "WebKit", "AppKit"]

hiddenimports += collect_submodules("statsforecast")

# --------------------------------------------------------------------------
# Excludes
#
# The old spec had excludes=[] while torch and transformers were installed,
# so ~1 GB of ML libraries were swept into a build whose code could not even
# import them. Being explicit keeps the Standard edition genuinely small.
# --------------------------------------------------------------------------

COMMON_EXCLUDES = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest",
    "sphinx", "setuptools._distutils", "PIL.ImageQt", "PyQt5", "PySide2",
]

AI_ONLY_MODULES = ["torch", "transformers", "timesfm", "chronos", "huggingface_hub",
                   "tokenizers", "safetensors", "accelerate"]

if IS_AI:
    excludes = list(COMMON_EXCLUDES)
    hiddenimports += ["torch", "transformers", "timesfm", "chronos", "huggingface_hub"]
    try:
        from PyInstaller.utils.hooks import copy_metadata

        # torch and transformers read their own package metadata at import time.
        for pkg in ("torch", "transformers", "tqdm", "regex", "filelock",
                    "requests", "packaging", "numpy", "huggingface-hub"):
            try:
                datas += copy_metadata(pkg)
            except Exception:
                pass
    except ImportError:
        pass
else:
    excludes = COMMON_EXCLUDES + AI_ONLY_MODULES

# --------------------------------------------------------------------------

block_cipher = None

a = Analysis(
    [os.path.join(ROOT, "shell", "app.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

ICON = None
for candidate in (os.path.join(SPECPATH, "icon.ico"), os.path.join(SPECPATH, "icon.icns")):
    if os.path.exists(candidate):
        if candidate.endswith(".ico") and IS_WINDOWS:
            ICON = candidate
        elif candidate.endswith(".icns") and IS_MAC:
            ICON = candidate

# One-dir on every platform — see the module docstring for why one-file is the
# wrong trade at this bundle size. Set FB_ONEFILE=1 to override.
ONE_FILE = os.environ.get("FB_ONEFILE") == "1"

if ONE_FILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=ICON,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        # "_internal" is PyInstaller's default, and it reads to a non-technical
        # user as either a mistake or something they were not meant to see —
        # 741 MB of it, sitting next to the one file they are told to open.
        # The name is cosmetic: sys._MEIPASS still points here either way.
        contents_directory="app-files",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=ICON,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )

    if IS_MAC:
        app = BUNDLE(
            coll,
            name=f"{APP_DISPLAY_NAME}.app",
            icon=ICON,
            bundle_identifier="com.kevinxing.forecastingbench",
            version=VERSION,
            info_plist={
                "CFBundleName": APP_DISPLAY_NAME,
                "CFBundleDisplayName": APP_DISPLAY_NAME,
                "CFBundleShortVersionString": VERSION,
                "CFBundleVersion": VERSION,
                "NSHighResolutionCapable": True,
                "NSHumanReadableCopyright": "Copyright (c) 2026 Kevin Xing. Apache-2.0.",
                # No network entitlement claims: the Standard edition never
                # makes an outbound request, and the privacy note says so.
                "LSMinimumSystemVersion": "10.15",
            },
        )
