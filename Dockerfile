FROM python:3.10-slim
WORKDIR /backend_file
COPY . /backend_file
RUN pip install -r requirements.txt
EXPOSE 8005
CMD ["python", "backend_file.py"]