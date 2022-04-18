# Use ubuntu base image
FROM ubuntu:16.04

# Install Python requirements
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# Install app dependencies from requirements.txt
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

# Copy 
COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "flask/app.py" ]