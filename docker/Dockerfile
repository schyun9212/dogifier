FROM nvcr.io/nvidia/pytorch:21.02-py3

ARG TZ="Asia/Seoul"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install -y vim zsh tmux git tzdata
RUN chsh -s $(which zsh)
ENTRYPOINT ["zsh"]

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

RUN python -m pip install --upgrade pip
RUN pip install regex==2018.01.10 msgpack==0.5.6
RUN pip install pytorch-lightning==1.4.0\
                yacs==0.1.7\
                timm==0.4.12\
                numpy==1.19.2\
                hydra-core==1.1.0

ARG PROJECT_NAME="project"
RUN mkdir /home/appuser/${PROJECT_NAME}
WORKDIR /home/appuser/${PROJECT_NAME}
