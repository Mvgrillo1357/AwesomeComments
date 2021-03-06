# Use Python slim base image
FROM python:3.8-slim

# Install app dependencies from requirements.txt
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy app
COPY . .

# Run app
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]