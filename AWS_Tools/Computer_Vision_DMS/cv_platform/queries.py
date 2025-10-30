# -*- coding: utf-8 -*-
"""
Queries
"""

import boto3

import boto3

def get_log_group_name(stack_name: str = "CvdmsStorageStack") -> str:
    cf = boto3.client("cloudformation")
    try:
        resp = cf.describe_stacks(StackName=stack_name)
    except cf.exceptions.ClientError as e:
        raise RuntimeError(f"Could not find stack {stack_name}: {e}")

    outputs = resp["Stacks"][0].get("Outputs", [])
    for o in outputs:
        if o["OutputKey"] == "AppLogGroupName":
            return o["OutputValue"]

    raise RuntimeError(f"Output 'AppLogGroupName' not found in stack {stack_name}")

