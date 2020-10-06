FROM ubuntu:latest

LABEL maintainer="Stanislav Schmidt <stanislav.schmidt@epfl.ch>"
LABEL version="1.0"
LABEL description="GROBID Quantities Server"

# ENV HTTP_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV HTTPS_PROXY='http://bbpproxy.epfl.ch:80/'
# ENV http_proxy='http://bbpproxy.epfl.ch:80/'
# ENV https_proxy='http://bbpproxy.epfl.ch:80/'


# Install java, git, unzip and wget
RUN apt-get update && apt-get install -y \
	default-jre \
	git \
	unzip \
	wget

# Add a user
RUN useradd --create-home grobiduser
WORKDIR /home/grobiduser
USER grobiduser

# Download and install GROBID
RUN true \
	&& git clone --depth=1 https://github.com/kermitt2/grobid.git grobid \
	&& cd grobid \
#	&& echo "systemProp.https.proxyHost=bbpproxy.epfl.ch" >> gradle.properties \
	&& ./gradlew clean install

# Download and install GROBID Quantities
RUN true \
	&& git clone --depth=1 https://github.com/kermitt2/grobid-quantities.git grobid/grobid-quantities \
	&& cd grobid/grobid-quantities/ \
#	&& echo "\nsystemProp.https.proxyHost=bbpproxy.epfl.ch" >> gradle.properties \
	&& ./gradlew copyModels \
	&& ./gradlew clean install

# Expose a port and set working directory
EXPOSE 8060
WORKDIR /home/grobiduser/grobid/grobid-quantities

ENTRYPOINT exec java -jar $(find build/libs -name "grobid-*onejar.jar") server resources/config/config.yml
# ENTRYPOINT exec java -jar build/libs/grobid-quantities-0.6.1-SNAPSHOT-onejar.jar server resources/config/config.yml
