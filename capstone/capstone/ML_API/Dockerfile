FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 5000

WORKDIR /code/app

RUN useradd -u 1001 testuser
USER testuser

# ENV FLASK_APP=app.server:app

# CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]

CMD exec gunicorn --bind :5000 --workers 1 --threads 8 --timeout 0 main:gunicorn_app
