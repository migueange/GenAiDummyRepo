import json

def transform_data(data):
    # Dummy transformation: capitalize all string values in the dictionary
    transformed = {}
    for key, value in data.items():
        if isinstance(value, str):
            transformed[key] = value.upper()
        else:
            transformed[key] = value
    return transformed

if __name__ == "__main__":
    sample_data = {
        "name": "john doe",
        "age": 30,
        "city": "new york"
    }
    print("Original data:", sample_data)
    transformed = transform_data(sample_data)
    print("Transformed data:", transformed)