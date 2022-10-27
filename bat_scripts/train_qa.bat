call c:\anaconda3\Scripts\activate.bat allan_dev
cd c:\allan
call winrun.bat
python -W ignore training\tags_and_qa\script_train_tags_qa.py --task=get_qa
call bat_scripts\restart_gateway.bat