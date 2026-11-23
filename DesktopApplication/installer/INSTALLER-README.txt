KalapakTTS Desktop 0.1.0

Install
-------
This is a per-user Windows x64 installation. It includes a portable CPython
3.12.10 runtime, the local FastAPI server and static editor, and the offline
WebView2 Evergreen Standalone installer. No system Python or repository folder
is used by the installed app.

VoxCPM2 weights and the Vanna voice reference are deliberately separate. Copy
the complete KalapakTTS-model-pack folder to a local drive or USB drive, then
run install_model_pack.ps1 from that folder. SHA-256 is checked before and
after copying to %APPDATA%\KalapakTTS\models. A drive formatted as NTFS or
exFAT is required; the model file exceeds FAT32's 4 GiB file limit.

GPU requirements
----------------
The bundled runtime uses PyTorch CUDA 12.8. The minimum NVIDIA Windows driver
is 570.65. At startup the app checks nvidia-smi and blocks CUDA model loading
if the driver is too old or cannot be verified; CPU fallback remains available
but is much slower. This PC's measured driver was 595.79.

Offline and updates
------------------
The WebView2 Evergreen Standalone installer is included for disconnected
machines and is installed per-user only if the runtime is absent. Evergreen
WebView2 receives Microsoft security updates separately from this app. The
KalapakTTS app itself has no updater; update it by running a newer release's
installer.

Uninstall and data
------------------
The uninstaller asks whether to keep %APPDATA%\KalapakTTS user data and the
separately installed model files. If uninstalled silently, it keeps both by
default. Media stored outside %APPDATA%\KalapakTTS is not removed.

Unsigned installer
------------------
This installer is not code-signed. Windows SmartScreen may show "Windows
protected your PC" or an unknown-publisher warning. Only proceed if the
installer came from a trusted source; select More info and then Run anyway.
This project does not bypass or suppress SmartScreen.

Third-party notice
------------------
The bundled FFmpeg binary is a GPLv3-or-later full build. The exact source and
all statically linked dependency sources have not been assembled for public
redistribution. See licenses\FFMPEG-NOTICE.txt; do not publish or share this
installer until source and license obligations are reviewed and satisfied.
