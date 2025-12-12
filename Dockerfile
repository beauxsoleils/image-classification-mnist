
FROM python:3.13

# set working dir
WORKDIR .

COPY . .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt   

CMD ["python", "main.py", "--epochs 400"]

EXPOSE 6006
