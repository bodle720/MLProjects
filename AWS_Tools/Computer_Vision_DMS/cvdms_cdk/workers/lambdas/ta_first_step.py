import json
def handler(event, context):
    print('Received event in first step of state machine: ' + json.dumps(event, indent=2))
    return {'statusCode': 200, 'message': 'Hello from first stp in state machine'}