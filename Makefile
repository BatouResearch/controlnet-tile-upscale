all: samples

serve:
	cog run -p 5000 python -m cog.server.http

samples:
	python samples.py
