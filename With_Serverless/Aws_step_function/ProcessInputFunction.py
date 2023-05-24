import json


def lambda_handler(event, context):
    # Read input numbers from the event
    A = int(event['worker'])
    B = int(event['batch_number'])
    

    
    
    
    # Create a table of size B with the required input fields for compute_gradient
    table = []
    for i in range(B):
        input_fields = {
            'rank': A,
            'batch_rank': i,
            'dataset': 'mnist',
            'model_str': 'vgg11',
            'optimiser': 'sgd',
            'lr': 0.01,
            'loss': 'NLL'
        }
        table.append(input_fields)



    return table

