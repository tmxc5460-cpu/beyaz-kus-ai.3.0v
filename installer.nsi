; BEYAZ KUŞ AI Windows Installer Script
; Creator: Ödül Ensar Yılmaz

!define APPNAME "BEYAZ KUŞ AI"
!define VERSION "2.5.0"
!define PUBLISHER "Ödül Ensar Yılmaz"
!define URL "https://beyazkus.ai"

; Modern UI
!include "MUI2.nsh"

; General settings
Name "${APPNAME}"
OutFile "BEYAZ_KUS_AI_Setup.exe"
InstallDir "$PROGRAMFILES\${APPNAME}"
InstallDirRegKey HKLM "Software\${APPNAME}" "InstallPath"
RequestExecutionLevel admin

; Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "icon-512.png"
!define MUI_UNICON "icon-512.png"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "Turkish"
!insertmacro MUI_LANGUAGE "English"

; Installer sections
Section "Core Files" SecCore
  SectionIn RO
  
  SetOutPath "$INSTDIR"
  
  ; Main application files
  File "index_bagimsiz.html"
  File "manifest.json"
  File "service-worker.js"
  File "package.json"
  File "README.md"
  
  ; Icon files
  File "icon-192.png"
  File "icon-512.png"
  
  ; Create shortcuts
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\index_bagimsiz.html" "" "$INSTDIR\icon-512.png" 0
  CreateShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\index_bagimsiz.html" "" "$INSTDIR\icon-512.png" 0
  
  ; Registry entries
  WriteRegStr HKLM "Software\${APPNAME}" "InstallPath" "$INSTDIR"
  WriteRegStr HKLM "Software\${APPNAME}" "Version" "${VERSION}"
  WriteRegStr HKLM "Software\${APPNAME}" "Publisher" "${PUBLISHER}"
  
  ; Uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSION}"
SectionEnd

Section "Desktop Shortcut" SecDesktop
  CreateShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\index_bagimsiz.html" "" "$INSTDIR\icon-512.png" 0
SectionEnd

Section "Start Menu Entries" SecStartMenu
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\index_bagimsiz.html" "" "$INSTDIR\icon-512.png" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

; Uninstaller section
Section "Uninstall"
  Delete "$INSTDIR\Uninstall.exe"
  Delete "$INSTDIR\index_bagimsiz.html"
  Delete "$INSTDIR\manifest.json"
  Delete "$INSTDIR\service-worker.js"
  Delete "$INSTDIR\package.json"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\icon-192.png"
  Delete "$INSTDIR\icon-512.png"
  
  Delete "$DESKTOP\${APPNAME}.lnk"
  Delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
  Delete "$SMPROGRAMS\${APPNAME}\Uninstall.lnk"
  RMDir "$SMPROGRAMS\${APPNAME}"
  
  RMDir "$INSTDIR"
  
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
  DeleteRegKey HKLM "Software\${APPNAME}"
SectionEnd

; Component descriptions
LangString DESC_SecCore ${LANG_TURKISH} "Ana uygulama dosyaları"
LangString DESC_SecCore ${LANG_ENGLISH} "Main application files"

LangString DESC_SecDesktop ${LANG_TURKISH} "Masaüstü kısayolu oluştur"
LangString DESC_SecDesktop ${LANG_ENGLISH} "Create desktop shortcut"

LangString DESC_SecStartMenu ${LANG_TURKISH} "Start menüsü girdileri oluştur"
LangString DESC_SecStartMenu ${LANG_ENGLISH} "Create start menu entries"

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCore} $(DESC_SecCore)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} $(DESC_SecDesktop)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenu} $(DESC_SecStartMenu)
!insertmacro MUI_FUNCTION_DESCRIPTION_END
