call c:\anaconda3\Scripts\activate.bat allan_dev
cd C:\allan
call winrun.bat
python training\merge\train_ner.py
call bat_scripts\restart_gateway.bat