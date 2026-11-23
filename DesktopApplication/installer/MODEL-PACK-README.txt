KalapakTTS Desktop model pack

Copy this entire folder to a local drive or USB drive formatted as NTFS or exFAT.
The VoxCPM2 weights contain a file larger than 4 GiB, which FAT32 cannot store.

Close KalapakTTS Desktop before installing the pack. Run install_model_pack.ps1
from this folder in PowerShell. The script verifies the manifest and SHA-256
for every model and Vanna voice file before replacing the local model folder.
It installs into %APPDATA%\KalapakTTS\models; it does not install or launch the
desktop application.

VoxCPM2 is provided under Apache License 2.0; see MODEL-LICENSE-APACHE-2.0.txt.
The Vanna reference audio is a separate local asset. Confirm rights and obtain
the speaker's permission before sharing or redistributing it.

The app remains usable for local history and drafts while the model pack is
absent, but speech generation reports a clear missing-model message.
