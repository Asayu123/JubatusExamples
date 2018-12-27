FROM jubatus/jubatus:1.1.0

RUN useradd -m jubatus
USER jubatus
WORKDIR '/home/jubatus'

ENV CONFIG_PATH='/home/jubatus/config.json'
ADD ./config.json ${CONFIG_PATH}

CMD jubarecommender -f ${CONFIG_PATH}
