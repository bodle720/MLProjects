# -*- coding: utf-8 -*-
"""
Glue Stack
"""

from aws_cdk import (
    Stack,
    Duration,
    CustomResource,
)
from constructs import Construct
from aws_cdk.aws_lambda import Function, Runtime, Code
from aws_cdk.aws_iam import PolicyStatement
from aws_cdk.custom_resources import Provider

class GlueCatalogStack(Stack):
    def __init__(self, scope: Construct, id: str, *, datalake_bucket, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Lambda that runs Athena DDL at deploy time
        ddl_lambda = Function(
            self, "AthenaDDLRunner",
            runtime=Runtime.PYTHON_3_12,
            handler="ddl.handler",
            code=Code.from_asset("lambdas/athena_ddl"),
            timeout=Duration.minutes(5),
            environment={
                "DATALAKE_BUCKET": datalake_bucket.bucket_name
            }
        )

        # Permissions for Athena, Glue Catalog, and S3
        ddl_lambda.add_to_role_policy(PolicyStatement(
            actions=["athena:StartQueryExecution", "athena:GetQueryExecution"],
            resources=["*"]
        ))
        ddl_lambda.add_to_role_policy(PolicyStatement(
            actions=["glue:*"],
            resources=["*"]
        ))
        ddl_lambda.add_to_role_policy(PolicyStatement(
            actions=["s3:*"],
            resources=[datalake_bucket.bucket_arn, f"{datalake_bucket.bucket_arn}/*"]
        ))

        # Custom resource provider
        provider = Provider(
            self, "AthenaDDLProvider",
            on_event_handler=ddl_lambda
        )

        # Trigger Lambda at deploy
        CustomResource(
            self, "AthenaDDLResource",
            service_token=provider.service_token
        )



