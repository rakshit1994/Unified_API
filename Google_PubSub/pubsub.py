import os
from os.path import join, dirname
from dotenv import load_dotenv
from concurrent.futures import TimeoutError
import json
from google.cloud import pubsub_v1
from pydantic.fields import Undefined

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
topic_name = 'projects/{project_id}/topics/{topic}'.format(
        project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'),
        topic=os.environ.get('GOOGLE_PUBSUB_TOPIC'),  # Set this to something appropriate.       
    )


def push_pubsub(obj):
    # The same for the publisher, except that the "audience" claim needs to be adjusted
    obj=json.loads(obj)
    credentials_path = join(dirname(__file__), '{cred_file}'.format(cred_file=os.environ.get('CREDENTIALS_FILE')))
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    topic_name = 'projects/{project_id}/topics/{topic}'.format(
        project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'),
        topic=os.environ.get('GOOGLE_PUBSUB_TOPIC'),  # Set this to something appropriate.       
    )
    # publisher_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
    # credentials_pub = credentials.with_claims(audience=publisher_audience)
    publisher = pubsub_v1.PublisherClient()
    data = json.dumps(obj).encode("utf-8")
    future = publisher.publish(topic_name, data)
    print(f'published message id {future.result()}')
    return json.dumps(future.result())





def callback(message):
    global response_obj
    print(f'Received message: {message}')
    print(f'data: {message.data}')

    if message:
        # for key in message.attributes:
        #     value = message.attributes.get(key)
        #     print(f"{key}: {value}")
        print(message.data)
        response_obj=message.data
    else:
        response_obj="No message to display"
    message.ack()      
    #return message.data     



def pull():
    credentials_path = join(dirname(__file__), '{cred_file}'.format(cred_file=os.environ.get('CREDENTIALS_FILE')))
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    timeout = 5.0                                                                       # timeout in seconds

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path  = 'projects/{project_id}/subscriptions/{sub}'.format(
         project_id=os.environ.get('GOOGLE_CLOUD_PROJECT'),
         sub=os.environ.get('GOOGLE_PUBSUB_SUB'),
         )
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f'Listening for messages on {subscription_path}')

    with subscriber:                                                # wrap subscriber in a 'with' block to automatically call close() when done
        try:
            streaming_pull_future.result(timeout=timeout)
            streaming_pull_future.result() 
                                                                    # going without a timeout will wait & block indefinitely
        except TimeoutError:
            streaming_pull_future.cancel()                          # trigger the shutdown
            streaming_pull_future.result()                         # block until the shutdown is complete
        try:
            response_obj
        except NameError:
            return "No messages to display yet."
        else:
            return response_obj
            