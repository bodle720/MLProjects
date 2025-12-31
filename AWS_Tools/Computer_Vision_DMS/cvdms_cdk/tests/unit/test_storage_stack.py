import aws_cdk as cdk
from aws_cdk.assertions import Template
import pytest

from stacks.main_stacks.storage_stack import StorageStack

@pytest.fixture
def template():
    app = cdk.App()
    stack = StorageStack(app, "TestStack")
    return Template.from_stack(stack)

def test_file_bucket_has_temp_lifecycle(template):
    # Check that at least one lifecycle rule exists with prefix "temp/"
    resources = template.find_resources("AWS::S3::Bucket")
    found = False
    for res in resources.values():
        rules = res["Properties"].get("LifecycleConfiguration", {}).get("Rules", [])
        for rule in rules:
            if rule.get("Prefix") == "temp/":
                days = rule["ExpirationInDays"]
                assert days <= 30
                assert days >= 1
                found = True
    assert found, "Expected lifecycle rule for temp/ prefix"

def test_athena_results_lifecycle_rule_days(template):
    # Find all S3 buckets in the template
    resources = template.find_resources("AWS::S3::Bucket")
    found = False
    for res in resources.values():
        rules = res["Properties"].get("LifecycleConfiguration", {}).get("Rules", [])
        for rule in rules:
            # Look for the athena-results/ prefix
            if rule.get("Prefix") == "athena-results/":
                days = rule["ExpirationInDays"]
                assert 1 <= days <= 30, (
                    f"Expected athena-results/ expiration between 1 and 30 days, got {days}"
                )
                found = True

    assert found, "Expected lifecycle rule for athena-results/ prefix"

def test_at_least_two_buckets_created(template):
    resources = template.find_resources("AWS::S3::Bucket")
    assert len(resources) >= 2

def test_all_expected_dynamodb_tables_exist(template):
    # Grab all DynamoDB tables from the synthesized template
    resources = template.find_resources("AWS::DynamoDB::Table")

    # Logical IDs are the keys in this dict
    logical_ids = resources.keys()

    expected_ids = ["JobTable", "DatasetsTable", "Sha256LookupTable", "LockTable"]

    for expected in expected_ids:
        assert any(expected in logical_id for logical_id in logical_ids), \
            f"Expected a DynamoDB table with logical ID containing '{expected}'"

def test_custom_resource_invokes_lambda(template):
    resources = template.find_resources("Custom::AWS")
    assert resources, "Expected at least one AwsCustomResource in the template"

    found = False
    for res in resources.values():
        create_block = res["Properties"].get("Create")
        if not create_block:
            continue
        # create_block is a dict with Fn::Join
        if "Fn::Join" in create_block:
            pieces = create_block["Fn::Join"][1]  # the list of string parts
            joined = "".join(str(p) for p in pieces)
            if '"service":"Lambda"' in joined and '"action":"invoke"' in joined:
                found = True

    assert found, "Expected custom resource configured to invoke a Lambda"




