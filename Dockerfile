FROM python:3.9-slim

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

COPY ./webapp /app/webapp
EXPOSE 5000

# We need to change directory before running the app because model.onnx is inside webapp
WORKDIR /app/webapp
ENTRYPOINT ["python"]
CMD ["app.py"]