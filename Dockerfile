FROM nikolaik/python-nodejs:python3.11-nodejs16-slim

WORKDIR /app

COPY . .

RUN apt-get update -y && \
    apt-get install -y xsel && \
    pip install -r requirements.txt && pip cache purge && \
    cd react_frontend && npm install && npm install -g serve && cd .. 

# Flask
CMD ["sh", "launch_app.sh"]
EXPOSE 5601 3000 5602