import json

def lambda_handler(event, context):
    # Dummy logic: just echo the input event
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Lambda called successfully!',
            'input': event
        })
    }

# Example of calling the lambda handler locally
if __name__ == "__main__":
    dummy_event = {"key": "value"}
    dummy_context = None  # Context is not used in this dummy example
    response = lambda_handler(dummy_event, dummy_context)
    print(response)