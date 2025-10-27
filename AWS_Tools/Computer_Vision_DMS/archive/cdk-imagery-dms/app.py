"""
Main CDK entrypoint for bootstrapping the project.
"""

import aws_cdk as cdk

from storage_stack import StorageStack
from compute_stack import ComputeStack
from workflow_stack import WorkflowStack
from glue_catalog_stack import GlueCatalogStack

app = cdk.App()

# 1. Storage (buckets, tables, etc.)
storage = StorageStack(app, "StorageStack")

# 2. Compute (Batch, Lambdas, etc.)
compute = ComputeStack(app, "ComputeStack",
    regular_bucket=storage.regular_bucket,
    jobs_table=storage.jobs_table
)

# 3. Workflow (Step Functions, orchestration)
workflow = WorkflowStack(app, "WorkflowStack",
    compute=compute,
    storage=storage
)

# 4. Glue Data Catalog setup (Athena DDL)
glue_catalog = GlueCatalogStack(app, "GlueCatalogStack",
    datalake_bucket=storage.regular_bucket
)

app.synth()
