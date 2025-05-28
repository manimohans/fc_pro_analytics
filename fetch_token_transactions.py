import os
import json
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('BASESCAN_API_KEY')
ADDRESS = '0x0bdca19c9801bb484285362fd5dd0c94592c874c'
CONTRACT_ADDRESS = '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913'

def load_existing_data():
    """Load existing addresses and transactions"""
    existing_addresses = set()
    existing_tx_hashes = set()
    existing_transactions = []
    
    # Load existing addresses
    if os.path.exists('addresses.csv'):
        with open('addresses.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_addresses.add(row['address'].lower())
    
    # Load existing transactions
    if os.path.exists('erc20.json'):
        with open('erc20.json', 'r') as f:
            data = json.load(f)
            if data.get('result') and isinstance(data['result'], list):
                existing_transactions = data['result']
                existing_tx_hashes = {tx['hash'] for tx in existing_transactions}
    
    return existing_addresses, existing_tx_hashes, existing_transactions

def fetch_token_transactions():
    # Load existing data
    existing_addresses, existing_tx_hashes, existing_transactions = load_existing_data()
    print(f"Loaded {len(existing_addresses)} existing addresses and {len(existing_transactions)} existing transactions")
    
    base_url = 'https://api.basescan.org/api'
    
    params = {
        'module': 'account',
        'action': 'tokentx',
        'address': ADDRESS,
        'contractaddress': CONTRACT_ADDRESS,
        'startblock': '0',
        'endblock': '99999999',
        'page': '1',
        'offset': '10000',
        'sort': 'desc',
        'apikey': API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"API Status: {data.get('status')}")
        print(f"API Message: {data.get('message')}")
        
        if data.get('result') and isinstance(data['result'], list):
            fetched_transactions = data['result']
            
            # Filter out transactions we already have
            new_transactions = [tx for tx in fetched_transactions if tx['hash'] not in existing_tx_hashes]
            
            print(f"\n=== Fetch Results ===")
            print(f"Fetched transactions: {len(fetched_transactions)}")
            print(f"New transactions: {len(new_transactions)}")
            print(f"Duplicate transactions: {len(fetched_transactions) - len(new_transactions)}")
            
            # Combine existing and new transactions
            all_transactions = existing_transactions + new_transactions
            
            # Save updated transactions to JSON
            updated_data = {
                'status': '1',
                'message': 'OK',
                'result': all_transactions
            }
            
            with open('erc20.json', 'w') as f:
                json.dump(updated_data, f, indent=2)
            
            print(f"\nTotal transactions in erc20.json: {len(all_transactions)}")
            
            if new_transactions:
                # Analytics for new transactions
                print(f"\n=== New Transaction Analytics ===")
                timestamps = [int(tx['timeStamp']) for tx in new_transactions]
                earliest_ts = min(timestamps)
                latest_ts = max(timestamps)
                
                # Convert to PST (UTC-8)
                earliest_dt = datetime.fromtimestamp(earliest_ts)
                latest_dt = datetime.fromtimestamp(latest_ts)
                
                # Format as PST (subtract 8 hours)
                from datetime import timedelta
                pst_offset = timedelta(hours=8)
                earliest_pst = earliest_dt - pst_offset
                latest_pst = latest_dt - pst_offset
                
                print(f"Earliest new transaction: {earliest_pst.strftime('%Y-%m-%d %H:%M:%S')} PST")
                print(f"Latest new transaction: {latest_pst.strftime('%Y-%m-%d %H:%M:%S')} PST")
                
                # Calculate total value for new transactions
                new_value = sum(int(tx['value']) for tx in new_transactions)
                new_usdc = new_value / 1_000_000
                print(f"New USDC transferred: ${new_usdc:,.2f}")
                
                # Extract new unique addresses
                new_from_addresses = set(tx['from'].lower() for tx in new_transactions) - existing_addresses
                
                if new_from_addresses:
                    # Append new addresses to CSV
                    csv_filename = 'addresses.csv'
                    file_exists = os.path.exists(csv_filename)
                    
                    with open(csv_filename, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not file_exists:
                            writer.writerow(['address'])  # Header if new file
                        for address in new_from_addresses:
                            writer.writerow([address])
                    
                    print(f"\nNew unique addresses: {len(new_from_addresses)}")
                    print(f"Appended to {csv_filename}")
                else:
                    print(f"\nNo new unique addresses found")
                
                # Overall statistics
                all_addresses = existing_addresses.union(set(tx['from'].lower() for tx in new_transactions))
                print(f"\nTotal unique addresses: {len(all_addresses)}")
            else:
                print("\nNo new transactions found. All fetched transactions already exist in the database.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    fetch_token_transactions()