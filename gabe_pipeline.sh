make install
make apple-shim
.venv/bin/python3 ingest.py

.venv/bin/python3 annotate.py --quality hybrid --all
.venv/bin/python3 annotate.py --terms --quality hybrid

.venv/bin/python3 topics.py

make nudges
make drift
make voiceprint
.venv/bin/python3 conflict_detect.py --max-label-turns 10 --label-top 50 --budget-minutes 20