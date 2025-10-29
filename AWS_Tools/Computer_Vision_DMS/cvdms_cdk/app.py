#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stacks.storage_stack import StorageStack


app = cdk.App()
StorageStack(app, "CvdmsStorageStack")

app.synth()
