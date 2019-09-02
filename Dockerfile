FROM python:3.7.4
LABEL maintainer="cthoyt@gmail.com"

# Maitain the list of requirements to make
#  docker builds go much faster, since these
#  do not change
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app
WORKDIR /app
RUN pip install -e .
