FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
	libxext6 \
	libfontconfig1 \
	libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
	libturbojpeg \
	software-properties-common \
	vim \
	screen \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD mkdir -p /workspace

RUN pip install albumentations
RUN pip install cnn_finetune
RUN pip install efficientnet-pytorch
RUN pip install timm
RUN pip install click
RUN pip install wandb
RUN pip install pyarrow
RUN pip install lz4
RUN pip install iterative-stratification
RUN pip install jupyterlab
RUN pip install SimpleITK
RUN pip install segmentation-models-pytorch
RUN pip install git+https://github.com/ngxbac/catalyst@ss --upgrade
#RUN pip install package

WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /workspace