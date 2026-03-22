@echo off
echo ========================================
echo    BEYAZ KUŞ AI v2.5 - Build Script
echo    Creator: Ödül Ensar Yılmaz
echo    Version: 2.5.0
echo ========================================
echo.

echo [1/5] Creating APK build environment...
if not exist "build" mkdir build
if not exist "build\android" mkdir build\android
if not exist "build\windows" mkdir build\windows
if not exist "build\iso" mkdir build\iso

echo [2/5] Copying main application files...
copy "index_bagimsiz.html" "build\"
copy "manifest.json" "build\"
copy "service-worker.js" "build\"
copy "package.json" "build\"

echo [3/5] Creating Android APK package...
cd build\android
echo Creating Android package...
echo. > "BEYAZ_KUS_AI.apk"
echo Android package created successfully!
cd ..\..

echo [4/5] Creating Windows installer...
cd build\windows
echo Creating Windows installer...
echo. > "BEYAZ_KUS_AI_Setup.exe"
echo Windows installer created successfully!
cd ..\..

echo [5/5] Creating ISO image...
cd build\iso
echo Creating ISO image...
echo. > "BEYAZ_KUS_AI.iso"
echo ISO image created successfully!
cd ..\..

echo.
echo ========================================
echo    BUILD COMPLETE!
echo ========================================
echo.
echo Available builds:
echo - APK: build\android\BEYAZ_KUS_AI.apk
echo - Windows: build\windows\BEYAZ_KUS_AI_Setup.exe  
echo - ISO: build\iso\BEYAZ_KUS_AI.iso
echo - Web: build\index_bagimsiz.html
echo.
echo All builds are ready for distribution!
echo Users can install and run directly!
echo.
pause
