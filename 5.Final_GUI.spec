# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['5.Final_GUI.py'],
    pathex=[],
    binaries=[],
    datas=[('classifier_model.tflite', '.'), ('regression_model.tflite', '.'), ('standard_scaler.pkl', '.'), ('ARS.png', '.'), ('efficientnetb3_notop.h5', '.')],
    hiddenimports=['tensorflow', 'customtkinter', 'umap', 'umaplearn', 'joblib', 'pandas', 'numpy', 'PIL', 'PIL.Image', 'PIL.ImageTk', 'PIL.ImageOps', 'cv2', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.simpledialog', 'sklearn', 'sklearn.decomposition', 'skimage', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'openpyxl', 'openpyxl.cell', 'openpyxl.cell._writer', 'tensorflow.keras.preprocessing', 'tensorflow.keras.applications.efficientnet'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='5.Final_GUI',
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
)
