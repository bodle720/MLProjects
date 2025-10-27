# -*- coding: utf-8 -*-
"""
Datalake Stack
"""

from aws_cdk import Stack, RemovalPolicy
from constructs import Construct
from aws_cdk.aws_glue import Database
from aws_cdk.aws_athena import CfnWorkGroup
from aws_cdk.aws_s3 import Bucket

class DataLakeStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        glue_db = Database(self, "GlueDatabase", database_name="cv_datalake")

        results_bucket = Bucket(self, "AthenaResultsBucket",
                                removal_policy=RemovalPolicy.DESTROY,
                                auto_delete_objects=True)

        wg = CfnWorkGroup(self, "AthenaWG",
                          name="cv_wg",
                          work_group_configuration=CfnWorkGroup.WorkGroupConfigurationProperty(
                              enforce_work_group_configuration=True,
                              result_configuration=CfnWorkGroup.ResultConfigurationProperty(
                                  output_location=f"s3://{results_bucket.bucket_name}/results/"
                              )
                          ))

        self.outputs = {
            "glueDbName": glue_db.database_name,
            "athenaWorkgroup": wg.name,
            "athenaResultsBucketName": results_bucket.bucket_name
        }


