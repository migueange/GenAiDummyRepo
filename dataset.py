import pandas as pd
from faker import Faker
import random

def create_dummy_dataset(num_rows=100):
    fake = Faker()
    data = []

    for _ in range(num_rows):
        row = {
            'id': fake.uuid4(),
            'name': fake.name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'address': fake.address(),
            'dob': fake.date_of_birth(minimum_age=18, maximum_age=80),
            'company': fake.company(),
            'job_title': fake.job(),
            'salary': round(random.uniform(30000, 120000), 2),
            'is_active': random.choice([True, False])
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df

# Example usage
if __name__ == "__main__":
    df = create_dummy_dataset(num_rows=50)
    print(df.head())
    df.to_csv("dummy_dataset.csv", index=False)
