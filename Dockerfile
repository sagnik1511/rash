FROM ubuntu:latest

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y vim build-essential git cmake net-tools gdb clang libopenblas-dev

WORKDIR /rash-nn


COPY rash/ /rash-nn/rash
COPY tests/ /rash-nn/tests