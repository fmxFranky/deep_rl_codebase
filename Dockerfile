FROM tensorflow/tensorflow:nightly-gpu
################################
# Install apt-get Requirements #
################################
ENV LANG C.UTF-8
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --default-timeout 100"
ENV GIT_CLONE="git clone --depth 10"
RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    apt-utils build-essential ca-certificates dpkg-dev pkg-config software-properties-common module-init-tools \
    cifs-utils openssh-server nfs-common net-tools iputils-ping iproute2 locales htop tzdata \
    tar wget git swig vim curl tmux zip unzip rar unrar \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev libglfw3 libglew2.0


################
# Set Timezone #
################
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    echo "Asia/Shanghai" > /etc/timezone && \
    rm -f /etc/localtime && \
    rm -rf /usr/share/zoneinfo/UTC && \
    dpkg-reconfigure --frontend=noninteractive tzdata

##########
# Mujoco #
##########
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_KEY_PATH /root/.mujoco${MUJOCO_KEY_PATH}

################
# RL Framework #
################
WORKDIR /root/src/
RUN $GIT_CLONE https://github.com/deepmind/acme.git && \
    cd acme && python setup.py install --nightly
RUN $PIP_INSTALL pip wandb mlflow boto3 seaborn runx \
    ray[debug,tune,rllib,dashboard] gym procgen atari-py opencv-python \
    dm-reverb-nightly tfp-nightly dm-sonnet trfl \
    'bsuite @ git+git://github.com/deepmind/bsuite.git#egg=bsuite' dm-control gym[atari] \
    jax jaxlib dm-haiku dataclasses 'rlax @ git+git://github.com/deepmind/rlax.git#egg=rlax'
RUN pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html

#############
# Oh my zsh #
#############
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/deluan/zsh-in-docker/master/zsh-in-docker.sh)" -- \
    -t ys \
    -p git \
    -p history \
    -p history-substring-search \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting

##################
# Apt auto clean #
##################
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache/pip
WORKDIR /root
SHELL ["/bin/zsh"]