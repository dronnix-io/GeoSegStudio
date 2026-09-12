# Chapter 2 — Installing GeoSeg Studio the Right Way

---

### 2.1 What Gets Installed and Where

Before touching the installation steps, it helps to understand exactly what GeoSeg Studio installs and where it puts things. This prevents confusion and makes troubleshooting much easier.

GeoSeg Studio is a QGIS plugin — a folder of Python files that QGIS loads at startup. The plugin itself is lightweight: a few hundred kilobytes of code. What makes it large is the deep learning framework it depends on: **PyTorch**.

PyTorch cannot be installed into QGIS's own Python environment without risking conflicts with QGIS's own dependencies. GeoSeg Studio solves this by installing PyTorch into a completely **isolated virtual environment** inside the plugin folder itself, at:

```
<QGIS plugins folder>/GeoSegStudio/env/
```

This environment is invisible to QGIS and to your system Python. It does not affect anything outside the plugin folder. If you ever want to remove it, you delete the `env/` folder and it is gone.

The total disk space required is approximately **5 GB** for a GPU (CUDA) installation, or **2.5 GB** for CPU-only. Make sure you have this space available before starting.

---

### 2.2 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Operating System | Windows 10 | Windows 11 |
| QGIS | 3.34 | 3.38 or later |
| Python | 3.9 (bundled with QGIS) | 3.11 |
| RAM | 8 GB | 16 GB or more |
| Disk space | 5 GB free | 10 GB free |
| GPU (optional) | NVIDIA with CUDA 11.8 | NVIDIA RTX series with CUDA 12.4 |

**Operating system note:** GeoSeg Studio currently runs on Windows only. Linux and macOS support is planned for a future release. If you are on Linux or macOS, the plugin will install but the PyTorch environment setup will not complete correctly.

**QGIS version note:** QGIS 3.34 is the minimum supported version. Earlier versions use an older Python runtime that is not compatible. If you are running an older QGIS, update it first — QGIS updates are free and non-destructive.

---

### 2.3 Step 1 — Install the Plugin

You have three options. Option A is recommended for most users.

---

**Option A: QGIS Plugin Manager (recommended)**

1. Open QGIS
2. Go to **Plugins → Manage and Install Plugins**
3. Make sure the **All** tab is selected at the top
4. In the search box, type `GeoSeg Studio`
5. Click on the result, then click **Install Plugin**

QGIS will download and install the plugin automatically. When it completes, you will see a new icon in your QGIS toolbar and a new entry under **Plugins → GeoSeg Studio**.

---

**Option B: Install from ZIP**

Use this option if you downloaded a specific release version, or if your QGIS does not have internet access.

1. Download `GeoSegStudio.zip` from the [Releases page](https://github.com/dronnix-io/GeoSegStudio/releases)
2. In QGIS go to **Plugins → Manage and Install Plugins → Install from ZIP**
3. Browse to the downloaded ZIP file
4. Click **Install Plugin**

---

**Option C: Build from source**

Use this option if you want the latest development version or if you plan to modify the plugin.

```bash
git clone https://github.com/dronnix-io/GeoSegStudio.git
cd GeoSegStudio
python package.py
```

This produces `dist/GeoSegStudio.zip`. Install it using Option B above.

---

### 2.4 Step 2 — Launch the Plugin

After installation, open GeoSeg Studio via:

**Plugins → GeoSeg Studio → GeoSeg Studio**

Alternatively, click the **GeoSeg Studio icon** in the QGIS toolbar. The plugin opens as a dock panel on the right side of the QGIS window.

On the very first launch, a setup dialog will appear asking you to choose your hardware. Do not skip this step — it determines which version of PyTorch gets installed.

---

### 2.5 Step 3 — Choose Your Hardware and Install PyTorch

The setup dialog presents four options:

| Option | When to choose it |
|--------|------------------|
| NVIDIA GPU — CUDA 11.8 | You have an NVIDIA GTX or RTX 10xx, 20xx, or 30xx series GPU with older drivers |
| NVIDIA GPU — CUDA 12.1 | You have an NVIDIA RTX 30xx or 40xx series GPU |
| NVIDIA GPU — CUDA 12.4 | You have an NVIDIA RTX 40xx series GPU with the latest drivers |
| CPU only | You have no NVIDIA GPU, or you are unsure |

**How to check your CUDA version:**
Open a terminal (Windows key → type `cmd` → Enter) and run:
```
nvidia-smi
```
Your CUDA version is shown in the top-right corner of the output. If `nvidia-smi` is not found, you either have no NVIDIA GPU or the drivers are not installed — select CPU only.

**AMD and Intel GPUs** are not supported for GPU acceleration. Select CPU only. The plugin will work fully, just without GPU speed.

**Not sure? Select CPU only.** It is always safe. You can reinstall later by deleting the `env/` folder inside the plugin directory and relaunching QGIS.

After you click **Install**, the plugin downloads and installs PyTorch into the isolated environment. This takes 5–15 minutes depending on your internet connection. A progress bar shows the download status. Do not close QGIS during this step.

When the installation completes, the setup dialog closes and the plugin is ready to use.

---

### 2.6 Verifying Your Setup

Before working on real data, verify that everything installed correctly.

**Check 1 — Plugin loads without errors**
Close and reopen QGIS. Open the plugin. If you see the four-tab interface (Prepare, Train, Evaluate, Predict), the plugin loaded correctly.

**Check 2 — PyTorch is available**
In the Train tab, open the device selector. If you installed a CUDA version and your GPU is correctly detected, you will see your GPU listed (for example: `CUDA — NVIDIA GeForce RTX 3080`). If you installed CPU only, you will see `CPU`.

**Check 3 — A quick training run**
Load any small raster and vector layer into QGIS. Run the Prepare tab with default settings on a small area. If tiles are generated without errors, the core pipeline is working.

---

### 2.7 Common Installation Failures and How to Fix Them

**Problem: Plugin Manager shows "GeoSeg Studio" but install button is greyed out**
Your QGIS version is below 3.34. Update QGIS from qgis.org.

---

**Problem: PyTorch installation hangs or freezes at a certain percentage**
This is almost always a network timeout. The PyTorch packages are large (1–3 GB). On a slow connection, the download can time out silently.

Fix: Delete the partially installed `env/` folder inside the plugin directory, then relaunch QGIS and try again. On a slow connection, use a wired Ethernet connection rather than Wi-Fi.

---

**Problem: Setup dialog says "Installation failed" with a pip error**
This usually means a disk space issue or a permissions issue.

- Check that you have at least 5 GB free on the drive where QGIS plugins are installed
- Make sure QGIS is not running as Administrator (counterintuitively, running as Admin can cause permission issues with virtual environments on Windows)

---

**Problem: Plugin loads but the device selector only shows CPU, even though I have an NVIDIA GPU**
Several possible causes:

1. You selected CPU only during setup. Fix: delete `env/` and reinstall with the correct CUDA version.
2. Your CUDA version selection does not match your driver version. Fix: run `nvidia-smi` to confirm your CUDA version, delete `env/`, and reinstall with the matching option.
3. Your NVIDIA drivers are outdated. Fix: update drivers from nvidia.com, then reinstall the plugin environment.

---

**Problem: QGIS crashes on startup after installing the plugin**
Rarely, a partial install can leave a corrupted environment that crashes QGIS on load.

Fix:
1. Disable the plugin via **Plugins → Manage and Install Plugins** (uncheck GeoSeg Studio)
2. Delete the `env/` folder inside the plugin directory
3. Re-enable the plugin
4. Let the setup dialog run again

---

**Problem: "DLL load failed" error on Windows when running training**
This is a CUDA/cuDNN compatibility error. Your selected CUDA version does not match your installed NVIDIA drivers.

Fix: Update your NVIDIA drivers to the latest version from nvidia.com. If the error persists, try reinstalling with CUDA 12.1 instead of 12.4 (or vice versa), as driver/CUDA compatibility can be version-specific.

---

### 2.8 Finding the Plugin Directory

You may need to locate the plugin directory to delete the `env/` folder, examine log files, or perform a manual reinstall.

**Default location on Windows:**
```
C:\Users\<your username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\GeoSegStudio\
```

The fastest way to find it: in QGIS, go to **Settings → User Profiles → Open Active Profile Folder**. From there, navigate to `python/plugins/GeoSegStudio/`.

---

### 2.9 Updating the Plugin

When a new version is released, QGIS will notify you in the Plugin Manager with an "Upgrade" button. Click it to update. The `env/` folder is preserved during updates — you do not need to reinstall PyTorch unless the release notes specifically say otherwise.

---

*Next: Chapter 3 — Your Data, Your Rules: Preparing a Training Dataset*
