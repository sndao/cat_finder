found cats + photos:
https://docs.google.com/spreadsheets/d/1de7uuQjjZhu8ghZKagVy-zpUHpSyrMPvqni299O8AwI/edit?gid=1349108449#gid=1349108449

https://github.com/sndao/cat_finder/

https://share.gemini.google/LYD3RBTJW84K
 
https://share.gemini.google/YIVgp36SKsiz -- i asked it 're-draw garfield with emi's cat colors'..)

https://claude.ai/share/04881b66-c48e-41ae-babf-78640c4f316c


Schedule to run everyday:

# crontab -l 
0 3 * * * /Users/stevendao/cat_finder/venv/bin/python /Users/stevendao/cat_finder/cat_finder.py >>  /Users/stevendao/var/log/cron_cat_log-`date +\%Y-\%m-\%d`.log 2>&1
