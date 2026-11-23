#define AppName "KalapakTTS Desktop"
#define AppVersion "0.1.0"
#define AppPublisher "Kalapak Team"

[Setup]
AppId={{83BC6F05-B6A4-4887-B8FE-97FC1A89AC02}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
SetupIconFile=..\app\kalapaktts.ico
DefaultDirName={localappdata}\Programs\KalapakTTS
DefaultGroupName=KalapakTTS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0.17763
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\app\kalapaktts.ico
OutputBaseFilename=KalapakTTS-Desktop-Setup-{#AppVersion}
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\runtime\*"; DestDir: "{app}\runtime"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\app\*"; DestDir: "{app}\app"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\server\*"; DestDir: "{app}\server"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\ui\*"; DestDir: "{app}\ui"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "INSTALLER-README.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "FFMPEG-NOTICE.txt"; DestDir: "{app}\licenses"; Flags: ignoreversion
Source: "..\build\webview2\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not IsWebView2RuntimeInstalled

[Icons]
Name: "{autoprograms}\KalapakTTS Desktop"; Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\app\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\app\kalapaktts.ico"
Name: "{autodesktop}\KalapakTTS Desktop"; Filename: "{app}\runtime\python\pythonw.exe"; Parameters: """{app}\app\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\app\kalapaktts.ico"; Tasks: desktopicon

[Run]
Filename: "{tmp}\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"; Parameters: "/silent /install"; StatusMsg: "Installing the offline WebView2 Runtime..."; Flags: waituntilterminated runhidden; Check: not IsWebView2RuntimeInstalled

[Code]
const
  WebView2ClientId = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

var
  KeepUserData: Boolean;
  KeepModels: Boolean;

function RuntimeVersionExists(const RootKey: Integer; const SubKey: String): Boolean;
var
  VersionText: String;
begin
  Result :=
    RegQueryStringValue(RootKey, SubKey, 'pv', VersionText) and
    (VersionText <> '') and
    (VersionText <> '0.0.0.0');
end;

function IsWebView2RuntimeInstalled: Boolean;
var
  RuntimeKey: String;
begin
  RuntimeKey := 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + WebView2ClientId;
  Result :=
    RuntimeVersionExists(HKLM64, RuntimeKey) or
    RuntimeVersionExists(HKLM32, RuntimeKey) or
    RuntimeVersionExists(HKCU, RuntimeKey) or
    RuntimeVersionExists(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + WebView2ClientId);
end;

function InitializeUninstall: Boolean;
var
  ProfilePath: String;
  ModelsPath: String;
begin
  KeepUserData := True;
  KeepModels := True;
  ProfilePath := ExpandConstant('{userappdata}\KalapakTTS');
  ModelsPath := ProfilePath + '\models';
  if not UninstallSilent then
  begin
    if DirExists(ProfilePath) then
      KeepUserData :=
        MsgBox(
          'Keep local user data (history, drafts, settings, and audio) in ' +
          ProfilePath + '?',
          mbConfirmation, MB_YESNO) = IDYES;
    if DirExists(ModelsPath) then
      KeepModels :=
        MsgBox(
          'Keep the separately installed VoxCPM2 model files in ' +
          ModelsPath + '?',
          mbConfirmation, MB_YESNO) = IDYES;
  end;
  Result := True;
end;

procedure DeleteProfileEntriesExceptModels(const ProfilePath: String);
var
  SearchRecord: TFindRec;
  Entries: TArrayOfString;
  Count: Integer;
  EntryPath: String;
  Index: Integer;
begin
  Count := 0;
  if FindFirst(ProfilePath + '\*', SearchRecord) then
  begin
    try
      repeat
        if (SearchRecord.Name <> '.') and (SearchRecord.Name <> '..') and
          not (((SearchRecord.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0) and
          (CompareText(SearchRecord.Name, 'models') = 0)) then
        begin
          SetArrayLength(Entries, Count + 1);
          Entries[Count] := SearchRecord.Name;
          Count := Count + 1;
        end;
      until not FindNext(SearchRecord);
    finally
      FindClose(SearchRecord);
    end;
  end;

  for Index := 0 to Count - 1 do
  begin
    EntryPath := ProfilePath + '\' + Entries[Index];
    if DirExists(EntryPath) then
      DelTree(EntryPath, True, True, True)
    else if FileExists(EntryPath) then
      DeleteFile(EntryPath);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ProfilePath: String;
  ModelsPath: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    ProfilePath := ExpandConstant('{userappdata}\KalapakTTS');
    ModelsPath := ProfilePath + '\models';
    if not KeepUserData then
    begin
      if KeepModels and DirExists(ModelsPath) then
        DeleteProfileEntriesExceptModels(ProfilePath)
      else
        DelTree(ProfilePath, True, True, True);
    end
    else if not KeepModels then
      DelTree(ModelsPath, True, True, True);
  end;
end;
