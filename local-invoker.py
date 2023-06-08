from cloudevents.http import CloudEvent
from cloudevents.conversion import to_structured
import requests
import sys


# Create a cloudevent using https://github.com/cloudevents/sdk-python
# Note we only need source and type because the cloudevents constructor by
# default will set "specversion" to the most recent cloudevent version (e.g. 1.0)
# and "id" to a generated uuid.uuid4 string.
attributes = {
    "Content-Type": "application/json",
    "source": "from-galaxy-far-far-away",
    "type": "cloudevent.greet.you",
}
# data = {
#     "bucket": "cs23-ps414-images-bkt",
#     "contentType": "image/jpeg",
#     "crc32c": "74hFmg==",
#     "etag": "CMDogvqSsv8CEAE=",
#     "generation": "1686174888408128",
#     "id": "cs23-ps414-images-bkt/test/miner.jpg/1686174888408128",
#     "kind": "storage#object",
#     "md5Hash": "J1j9saiD0oopuyf4h1lJKw==",
#     "mediaLink": "https://storage.googleapis.com/download/storage/v1/b/cs23-ps414-images-bkt/o/test%2Fminer.jpg?generation=1686174888408128&alt=media",
#     "metageneration": "1",
#     "name": "images/037cc0cd-898d-486b-8e31-325e5e89696c",
#     "selfLink": "https://www.googleapis.com/storage/v1/b/cs23-ps414-images-bkt/o/test%2Fminer.jpg",
#     "size": "159269",
#     "storageClass": "STANDARD",
#     "timeCreated": "2023-06-07T21:54:48.419Z",
#     "timeStorageClassUpdated": "2023-06-07T21:54:48.419Z",
#     "updated": "2023-06-07T21:54:48.419Z"
# }
data = {
    "bucket": "cs23-ps414-images-bkt",
    "name": "images/" + sys.argv[1],
}

event = CloudEvent(attributes, data)

# Send the event to our local docker container listening on port 8080
headers, data = to_structured(event)
requests.post("http://localhost:8080/", headers=headers, data=data)