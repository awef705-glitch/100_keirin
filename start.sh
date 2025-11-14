#!/bin/bash
cd backend
PORT=${PORT:-10000}
exec gunicorn -w 1 -b 0.0.0.0:$PORT app:app --timeout 120
