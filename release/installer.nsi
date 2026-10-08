Unicode True
!include "MUI2.nsh"

!ifndef APP_VERSION
!define APP_VERSION "0.1.0"
!endif

Name "MI-BCI ${APP_VERSION}"
OutFile "..\dist\MI-BCI-Setup-${APP_VERSION}.exe"
InstallDir "D:\MI-BCI\app\${APP_VERSION}"
RequestExecutionLevel user
SetCompressor /SOLID lzma
ShowInstDetails show
ShowUninstDetails show

!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "SimpChinese"

Section "Install"
  SetOutPath "$INSTDIR\desktop"
  File /r "..\dist\MI-BCI\*"
  SetOutPath "$INSTDIR\mcp"
  File /r "..\dist\MI-BCI-MCP\*"
  SetOutPath "$INSTDIR"
  FileOpen $0 "$INSTDIR\mi-bci-installed.marker" w
  FileClose $0
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MI-BCI-${APP_VERSION}" "DisplayName" "MI-BCI ${APP_VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MI-BCI-${APP_VERSION}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MI-BCI-${APP_VERSION}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MI-BCI-${APP_VERSION}" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MI-BCI-${APP_VERSION}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  IfFileExists "$INSTDIR\mi-bci-installed.marker" +2 0
  Abort "Installation marker is missing; files were left untouched."
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MI-BCI-${APP_VERSION}"
  RMDir /r "$INSTDIR"
SectionEnd
