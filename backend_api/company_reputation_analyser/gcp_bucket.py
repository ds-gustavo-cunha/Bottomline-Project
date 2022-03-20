def upload_feedback_to_gcp(BUCKET_NAME, new_feedback):
    """It will append the new_feedback to the gcp bucket that holds all user feedbacks.

    Args
        BUCKET_NAME: a string with the bucket name on GPC storage for the project.
        new_feedback: a string with the new feedback to be appended.

    Return
        None: a none type object

    NOTE: we need to set the GOOGLE_APPLICATION_CREDENTIALS environment
        variable [export GOOGLE_APPLICATION_CREDENTIALS="[PATH-TO-JSON-CREDS]"]
        prior to use this function"""

    # import required libraries
    from google.cloud import storage

    # instanciate client object
    # it will use the GOOGLE_APPLICATION_CREDENTIAL environmental variable
    client = storage.Client()

    # define the required bucket
    bucket = client.bucket(BUCKET_NAME)

    # list the blob (binary large object)
    blobs = list(bucket.list_blobs(prefix="feedback/"))

    # define the desired blob to work on
    target_blob = blobs[0]

    # read the content of the blob: old feedbacks
    previous_feedbacks = target_blob.download_as_string().decode("utf-8")

    # define the final blob on GCP bucket
    target_blob = bucket.blob('feedback/feedback.txt')

    # open temporary feedback file
    with open("raw_data/tmp_feedbacks.txt", 'w') as f:
        # append new feedbacks to older ones
        # and write on temporary file
        f.write(f"{previous_feedbacks}\n{new_feedback}")

    # open temporary feedback file
    with open("raw_data/tmp_feedbacks.txt", 'r') as f:
        # upload feedback to GCP bucket
        target_blob.upload_from_file(f)

    return None
