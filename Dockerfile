FROM gitlab-registry.cern.ch/linuxsupport/cc7-base

MAINTAINER "Gabriel Moreira <gabriel.moreira@cern.ch>"

COPY packages packages

# Setting the ulimit to 1024 because of this bug in CC7: https://bugzilla.redhat.com/show_bug.cgi?id=1537564
# ADVICE: Always set the ulimit in CC7 docker if you are going to mess with yum commands in another shell
RUN ulimit -n 1024 && yum update -q -y \
    && yum install -y epel-release \
    && yum -y groupinstall "Development Tools" \
    && yum install -y $(cat packages) \
    && rm -f /packages

ARG USERNAME=hero
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

ENV HEP_SRCPATH=/home/$USERNAME/HEPHero
ENV HEP_OUTPATH=$HEP_SRCPATH/output
ENV REDIRECTOR=xrootd-cms.infn.it
ENV MACHINES=CERN
RUN echo 'alias hepenv="source $HEP_OUTPATH/hepenv_setup.sh"' >> /home/$USERNAME/.bashrc

ADD --chown=$USERNAME:$USERNAME . $HEP_SRCPATH

WORKDIR $HEP_OUTPATH
RUN wget 'https://cernbox.cern.ch/remote.php/dav/public-files/LNGQ6aDRQ9gzZNu/hepenv_setup.sh?access_token=null' -O hepenv_setup.sh
RUN wget 'https://cernbox.cern.ch/remote.php/dav/public-files/LNGQ6aDRQ9gzZNu/container_setup.sh?access_token=null' -O container_setup.sh
RUN wget 'https://cernbox.cern.ch/archiver?public-token=LNGQ6aDRQ9gzZNu&id=eoshome-g!188519735' -O libtorch_fix.tar
RUN tar xvf libtorch_fix.tar && rm libtorch_fix.tar

WORKDIR $HEP_SRCPATH
