Unicode True

OutFile "DeepFaceLivePortable.exe"

InstallDir $EXEDIR

!define TMP_DIR "$InstDir\__setup"
!define TMP_PYRUN_PATH "${TMP_DIR}\pyrun.cmd"
!define TMP_PYTHON_ZIP_PATH "${TMP_DIR}\python.zip"
!define TMP_PYTHON_PATH "${TMP_DIR}\python"
!define PIP_CACHE_DIR "${TMP_DIR}\pip_cache"

SilentInstall silent


Section


SetOutPath $InstDir
CreateDirectory "${TMP_DIR}"
CreateDirectory "${TMP_PYTHON_PATH}"

File "/oname=${TMP_PYRUN_PATH}" "pyrun.cmd" #

inetc::get "https://www.python.org/ftp/python/3.6.8/python-3.6.8-embed-amd64.zip" "${TMP_PYTHON_ZIP_PATH}"
Pop $0
StrCmp $0 "OK" +3
MessageBox MB_OK|MB_ICONEXCLAMATION "Python download error: $0, click OK to abort installation." IDOK
Abort

nsisunz::UnzipToLog "${TMP_PYTHON_ZIP_PATH}" "${TMP_PYTHON_PATH}"
Pop $0
StrCmp $0 "success" +3
MessageBox MB_OK|MB_ICONEXCLAMATION "Unzip error: $0, click OK to abort installation." IDOK
Abort

# Download installer script

File "/oname=${TMP_DIR}\WindowsBuilder.py" "..\WindowsBuilder.py" #


# https://github.com/iperov/DeepFaceLive/raw/master/


#ExecWait '"${TMP_PYRUN_PATH}" "${TMP_DIR}\WindowsBuilder.py" --build-type dfl-windows --python-ver 3.6.8 --release-dir "$InstDir\DeepFaceLive" --pip-cache-dir "${PIP_CACHE_DIR}" '

# RMDir /r "${TMP_DIR}"
# inetc::get https://raw.githubusercontent.com/iperov/DeepFaceLab/master/main.py $InstDir/asd.py
 
# MessageBox MB_OK "${NSIS_MAX_STRLEN}"

SectionEnd

