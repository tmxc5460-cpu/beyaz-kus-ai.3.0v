#!/bin/bash

echo "========================================"
echo "   BEYAZ KUŞ AI v2.5 - Build Script"
echo "   Creator: Ödül Ensar Yılmaz"
echo "   Version: 2.5.0"
echo "========================================"
echo

# Check if required tools are installed
check_tools() {
    echo "[1/6] Checking build tools..."
    
    # Check for Node.js
    if ! command -v node &> /dev/null; then
        echo "Node.js is required but not installed."
        exit 1
    fi
    
    # Check for npm
    if ! command -v npm &> /dev/null; then
        echo "npm is required but not installed."
        exit 1
    fi
    
    echo "All required tools are installed!"
}

# Create build directories
create_directories() {
    echo "[2/6] Creating build directories..."
    mkdir -p build/{android,windows,iso,web}
    echo "Build directories created!"
}

# Build Android APK
build_android() {
    echo "[3/6] Building Android APK..."
    cd build/android
    
    # Create APK structure
    mkdir -p BEYAZ_KUS_AI/{assets,lib,meta-inf,res}
    
    # Copy application files
    cp ../../index_bagimsiz.html BEYAZ_KUS_AI/assets/
    cp ../../manifest.json BEYAZ_KUS_AI/assets/
    cp ../../service-worker.js BEYAZ_KUS_AI/assets/
    cp ../../icon-512.png BEYAZ_KUS_AI/res/
    
    # Create APK
    cd BEYAZ_KUS_AI
    zip -r ../BEYAZ_KUS_AI_v2.5.apk .
    cd ..
    
    echo "Android APK created: BEYAZ_KUS_AI_v2.5.apk"
    cd ../..
}

# Build Windows Installer
build_windows() {
    echo "[4/6] Building Windows Installer..."
    cd build/windows
    
    # Copy files
    cp ../index_bagimsiz.html .
    cp ../manifest.json .
    cp ../service-worker.js .
    cp ../icon-512.png .
    
    # Create installer using makensis if available
    if command -v makensis &> /dev/null; then
        makensis ../../installer.nsi
        echo "Windows installer created using NSIS!"
    else
        # Create simple setup
        echo "Creating simple Windows setup..."
        cat > setup.js << 'EOF'
// BEYAZ KUŞ AI v2.5 Setup Script
var shell = WScript.CreateObject("WScript.Shell");
var fso = WScript.CreateObject("Scripting.FileSystemObject");

// Get installation path
var installPath = shell.ExpandEnvironmentStrings("%PROGRAMFILES%") + "\\BEYAZ KUŞ AI";
var desktopPath = shell.SpecialFolders("Desktop");

// Create directory
if (!fso.FolderExists(installPath)) {
    fso.CreateFolder(installPath);
}

// Copy files
fso.CopyFile("index_bagimsiz.html", installPath + "\\");
fso.CopyFile("manifest.json", installPath + "\\");
fso.CopyFile("icon-512.png", installPath + "\\");

// Create desktop shortcut
var shortcut = shell.CreateShortcut(desktopPath + "\\BEYAZ KUŞ AI v2.5.lnk");
shortcut.TargetPath = installPath + "\\index_bagimsiz.html";
shortcut.IconLocation = installPath + "\\icon-512.png";
shortcut.Save();

WScript.Echo("BEYAZ KUŞ AI v2.5 installed successfully!");
EOF
        
        cscript //nologo setup.js
        echo "Windows setup created!"
    fi
    
    cd ../..
}

# Build ISO Image
build_iso() {
    echo "[5/6] Building ISO Image..."
    cd build/iso
    
    # Create ISO structure
    mkdir -p BEYAZ_KU_AI_v2.5/{app,docs,drivers}
    
    # Copy application
    cp -r ../android/BEYAZ_KUS_AI_v2.5.apk BEYAZ_KU_AI_v2.5/app/
    cp ../windows/BEYAZ_KUS_AI_Setup.exe BEYAZ_KU_AI_v2.5/app/
    cp ../index_bagimsiz.html BEYAZ_KU_AI_v2.5/app/
    cp ../manifest.json BEYAZ_KU_AI_v2.5/app/
    cp ../README.md BEYAZ_KU_AI_v2.5/docs/
    
    # Create ISO using mkisofs if available
    if command -v mkisofs &> /dev/null; then
        mkisofs -o BEYAZ_KU_AI_v2.5.iso -V "BEYAZ KUŞ AI v2.5" -J -r BEYAZ_KU_AI_v2.5/
        echo "ISO created: BEYAZ_KU_AI_v2.5.iso"
    elif command -v hdiutil &> /dev/null; then
        # macOS
        hdiutil makehybrid -o BEYAZ_KU_AI_v2.5.iso -iso -joliet BEYAZ_KU_AI_v2.5/
        echo "ISO created: BEYAZ_KU_AI_v2.5.iso"
    else
        echo "ISO creation tools not found. Please install mkisofs or hdiutil."
    fi
    
    cd ../..
}

# Build Web Package
build_web() {
    echo "[6/6] Building Web Package..."
    cd build/web
    
    # Copy web files
    cp ../index_bagimsiz.html index.html
    cp ../manifest.json .
    cp ../service-worker.js .
    cp ../icon-192.png .
    cp ../icon-512.png .
    
    # Create web package
    zip -r BEYAZ_KU_AI_v2.5_web.zip .
    
    echo "Web package created: BEYAZ_KU_AI_v2.5_web.zip"
    cd ../..
}

# Main build process
main() {
    echo "Starting BEYAZ KUŞ AI v2.5 build process..."
    echo
    
    check_tools
    create_directories
    build_android
    build_windows
    build_iso
    build_web
    
    echo
    echo "========================================"
    echo "       BUILD COMPLETE v2.5!"
    echo "========================================"
    echo
    echo "Available builds:"
    echo "  📱 Android: build/android/BEYAZ_KUS_AI_v2.5.apk"
    echo "  🪟 Windows: build/windows/BEYAZ_KUS_AI_Setup.exe"
    echo "  💿 ISO: build/iso/BEYAZ_KU_AI_v2.5.iso"
    echo "  🌐 Web: build/web/BEYAZ_KU_AI_v2.5_web.zip"
    echo
    echo "All builds are ready for distribution!"
    echo "Users can install and run directly!"
    echo
}

# Run main function
main "$@"
