FROM  python:3.11
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt 
EXPOSE $PORT
# CMD waitress-serve --listen=127.0.0.1:$PORT app:app
# CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app
CMD flask run --host=0.0.0.0 --port=5000
