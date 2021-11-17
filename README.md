# Unified_API - VectorAI
 
## Ways to Run Application:
1. Clone/Pull from Github or Dockerhub to local system.
2. Place the .env file and the google-credentials.json (service credentials from GCloud) file in the Unified_API/Google_PubSub/ folder, then:
    * Run using python (version > 3.6):
        * 1. pip install -r requirements.txt
        * 2. python app.py
    * Run using docker:
        * Build docker image using Dockerfile and run the image as container
           * cd inside the Unified_API folder 
           * docker build -t imagename . 
           * docker run -dp 8000:8000 imagename 
3.  Go to localhost:8000 to access application.

### .env file contents:
* GOOGLE_CLOUD_PROJECT=<Prject-Name> Ex:vectorai-332109
* GOOGLE_PUBSUB_TOPIC=<Subsciber-Endpoint-Name> Ex: ML_Score
* GOOGLE_PUBSUB_SUB=<Subsciber-Endpoint-Name> Ex: ML_Score-sub
* CREDENTIALS_FILE=<credentials-file-name.json> Ex:vectorai-332109-55343e53a98d.json



## Relevant Info:
-- Train and load dataset from Unified_API/ML_Model/cnn_model.py file

-- Unified_API/app.py has the endpoints

-- Unified_API/Google_PubSub/vectorai-332109-55343e53a98d.json has the Google Service Credentials

-- HTML page (templates) are placed in Unified_API/templates folder.

-- Application default UI will be accessible at localhost:8000

-- Swagger API UI can be accessed at localhost:8000/docs
