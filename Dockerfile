
FROM python:3.13

# set working dir
WORKDIR .
COPY . .

CMD ["python", "main.py"]

RUN tensorboard --log_dir runs 

