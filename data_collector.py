import requests
import csv
import os
from datetime import datetime
import time

# URL of the Flask application's endpoint that provides live traffic data
url = "http://localhost:4000/firewalltraffic"  # Adjust the port if different

def get_traffic_data():
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.RequestException as e:
        print(f"Failed to fetch data: {e}")
        return None

def save_to_csv(data):
    if data is None:
        return

    # Create the 'csv' directory if it doesn't exist
    if not os.path.exists('csv'):
        os.makedirs('csv')

    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = os.path.join('csv', f"{date_str}.csv")
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file is new
            header = ["Timestamp"] + list(data.keys())
            writer.writerow(header)

        # Write the data row
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [timestamp] + list(data.values())
        writer.writerow(row)

def main():
    while True:
        traffic_data = get_traffic_data()
        save_to_csv(traffic_data)
        # Wait for a specified interval before fetching data again (e.g., every minute)
        time.sleep(60)

if __name__ == "__main__":
    main()
