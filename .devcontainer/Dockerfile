FROM cern/cc7-base

RUN yum -y install epel-release sudo git wget which R

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

ENV LCG_VIEW /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos7-gcc9-opt/setup.sh
ENV HEP_VENV /workspace/.hepenv

RUN echo "alias hepenv='source $LCG_VIEW && python -m venv $HEP_VENV && source $HEP_VENV/bin/activate'" >> /home/$USERNAME/.bashrc
RUN echo "alias ccm='rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile'" >> /home/$USERNAME/.bashrc