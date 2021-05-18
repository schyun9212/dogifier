FROM nvcr.io/nvidia/pytorch:20.07-py3

RUN apt update
RUN apt install -y vim zsh tmux git
RUN chsh -s $(which zsh)
ENTRYPOINT ["zsh"]

ARG PROJECT_NAME="project"

# create a non-root user
ARG USER="appuser"
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} ${USER} -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

RUN python -m pip install --upgrade pip
RUN pip install regex==2018.01.10 msgpack==0.5.6
RUN pip install torchvision\
                pytorch-lightning==1.0.0\
                hydra-core==1.0.4\
                timm==0.2.1\
                numpy==1.19.2\
                imgaug==0.4.0\
                pyarrow==2.0.0\
                pylint==2.6.0

RUN mkdir /home/${USER}/${PROJECT_NAME}
WORKDIR /home/${USER}/${PROJECT_NAME}
