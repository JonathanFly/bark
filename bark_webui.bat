@echo off
call %USERPROFILE%\mambaforge\Scripts\activate.bat bark-infinity-oneclick
python %USERPROFILE%\bark\bark_webui.py
pause
