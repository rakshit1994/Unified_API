FROM alpine:latest
FROM python:3.6

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

#COPY . /app

RUN chmod -R 777 /app

EXPOSE 8000

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]