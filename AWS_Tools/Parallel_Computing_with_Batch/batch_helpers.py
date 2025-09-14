# -*- coding: utf-8 -*-
"""
Helper functions for the main batch script.
"""

import time
from botocore.exceptions import ClientError

def wait_for_compute_env_valid(batch_client, env_name, timeout=300, interval=5):
    """
    Poll describe_compute_environments() until the compute environment's status is VALID
    or until timeout is reached.

    :param batch_client: boto3 Batch client
    :param env_name: Name or ARN of the compute environment
    :param timeout: Max seconds to wait
    :param interval: Seconds between polls
    :return: True if VALID within timeout, else False
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            resp = batch_client.describe_compute_environments(
                computeEnvironments=[env_name]
            )
            envs = resp.get("computeEnvironments", [])
            if envs:
                status = envs[0].get("status")
                reason = envs[0].get("statusReason", "")
                if status == "VALID":
                    return True
                print(f"CE {env_name} status: {status} ({reason}). Waiting {interval}s…")
            else:
                print(f"Compute environment {env_name} not found. Waiting {interval}s…")
        except ClientError as e:
            print(f"Error describing CE: {e}. Retrying in {interval}s…")

        time.sleep(interval)

    return False

def wait_for_queue_valid(batch_client, queue_name, timeout=300, interval=5):
    """
    Polls describe_job_queues() until the specified queue's status is VALID
    or until timeout is reached.

    :param batch_client: boto3 Batch client
    :param queue_name: Name or ARN of the job queue
    :param timeout: Maximum seconds to wait
    :param interval: Seconds between polls
    :return: True if queue is VALID within timeout, else False
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            response = batch_client.describe_job_queues(
                jobQueues=[queue_name]
            )
            queues = response.get("jobQueues", [])
            if queues:
                status = queues[0].get("status")
                if status == "VALID":
                    return True
                print(f"Queue {queue_name} status: {status}. Waiting {interval}s…")
            else:
                print(f"Queue {queue_name} not found. Waiting {interval}s…")
        except ClientError as e:
            print(f"Error describing queue: {e}. Retrying in {interval}s…")

        time.sleep(interval)

    return False