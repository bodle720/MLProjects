import aws_cdk as core
import aws_cdk.assertions as assertions

from stacks.storage_stack import CvdmsCdkStack

# example tests. To run these tests, uncomment this file along with the example
# resource in stacks/storage_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = CvdmsCdkStack(app, "cvdms-cdk")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
