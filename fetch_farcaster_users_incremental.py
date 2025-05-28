import os
import csv
import json
import requests
from dotenv import load_dotenv
import time

load_dotenv()

# Get API key from environment variables
NEYNAR_API_KEY = os.getenv("NEYNAR_API_KEY")
if not NEYNAR_API_KEY:
    raise ValueError("NEYNAR_API_KEY not found in environment variables")

def load_addresses():
    """Load addresses from addresses.csv"""
    addresses = []
    if os.path.exists('addresses.csv'):
        with open('addresses.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                addresses.append(row['address'])
    return addresses

def load_existing_farcaster_data():
    """Load existing Farcaster data to check which addresses we already have"""
    existing_addresses = set()
    existing_data = {}
    
    if os.path.exists('addresses_fc.csv'):
        with open('addresses_fc.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                address = row['address'].lower()
                existing_addresses.add(address)
                existing_data[address] = row
    
    return existing_addresses, existing_data

def fetch_users_batch(addresses_batch):
    """Fetch user data for a batch of addresses (max 100)"""
    url = "https://api.neynar.com/v2/farcaster/user/bulk-by-address"
    
    headers = {
        "x-api-key": NEYNAR_API_KEY,
        "x-neynar-experimental": "false"
    }
    
    # Join addresses with just comma (no space) to reduce URL length
    addresses_string = ",".join(addresses_batch)
    querystring = {"addresses": addresses_string}
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching batch: {e}")
        return {}

def extract_user_data(address, user_data):
    """Extract relevant fields from user data"""
    if not user_data:
        return {
            'address': address,
            'fid': '',
            'username': '',
            'display_name': '',
            'pfp_url': '',
            'bio': '',
            'latitude': '',
            'longitude': '',
            'city': '',
            'state': '',
            'state_code': '',
            'country': '',
            'country_code': '',
            'follower_count': '',
            'following_count': '',
            'verified_accounts': '',
            'score': ''
        }
    
    # Take the first user if multiple exist for an address
    user = user_data[0]
    
    # Extract bio text (properly escape for CSV)
    bio = ''
    if user.get('profile', {}).get('bio', {}).get('text'):
        # Replace newlines with spaces and properly escape for CSV
        bio = user['profile']['bio']['text'].replace('\n', ' ').replace('\r', ' ')
        # Remove any other problematic characters
        bio = ' '.join(bio.split())  # Normalize whitespace
    
    # Extract location data
    location = user.get('profile', {}).get('location', {})
    address_info = location.get('address', {})
    
    # Extract verified accounts
    verified_accounts = []
    for account in user.get('verified_accounts', []):
        verified_accounts.append(f"{account.get('platform', '')}:{account.get('username', '')}")
    verified_accounts_str = ' | '.join(verified_accounts)
    
    # Extract score (handle both old and new field names)
    score = user.get('score', '')
    if not score and user.get('experimental', {}).get('neynar_user_score'):
        score = user['experimental']['neynar_user_score']
    
    return {
        'address': address,
        'fid': user.get('fid', ''),
        'username': user.get('username', ''),
        'display_name': user.get('display_name', ''),
        'pfp_url': user.get('pfp_url', ''),
        'bio': bio,
        'latitude': location.get('latitude', ''),
        'longitude': location.get('longitude', ''),
        'city': address_info.get('city', ''),
        'state': address_info.get('state', ''),
        'state_code': address_info.get('state_code', ''),
        'country': address_info.get('country', ''),
        'country_code': address_info.get('country_code', ''),
        'follower_count': user.get('follower_count', ''),
        'following_count': user.get('following_count', ''),
        'verified_accounts': verified_accounts_str,
        'score': score
    }

def fetch_farcaster_users_incremental():
    """Main function to fetch only NEW Farcaster user data"""
    # Load all addresses
    all_addresses = load_addresses()
    print(f"Total addresses in addresses.csv: {len(all_addresses)}")
    
    # Load existing Farcaster data
    existing_addresses, existing_data = load_existing_farcaster_data()
    print(f"Existing Farcaster data for: {len(existing_addresses)} addresses")
    
    # Find new addresses that need to be fetched
    new_addresses = []
    for addr in all_addresses:
        if addr.lower() not in existing_addresses:
            new_addresses.append(addr)
    
    print(f"New addresses to fetch: {len(new_addresses)}")
    
    if not new_addresses:
        print("No new addresses to fetch. All addresses already have Farcaster data.")
        return
    
    # Process new addresses in batches
    batch_size = 100
    new_user_data = []
    
    for i in range(0, len(new_addresses), batch_size):
        batch = new_addresses[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, len(new_addresses))} of {len(new_addresses)} new addresses)")
        
        # Fetch data for this batch
        response_data = fetch_users_batch(batch)
        
        # Process each address in the batch
        for address in batch:
            # Neynar returns addresses in lowercase
            address_key = address.lower()
            user_data = response_data.get(address_key, [])
            
            extracted_data = extract_user_data(address, user_data)
            new_user_data.append(extracted_data)
        
        # Add a small delay to avoid rate limiting
        if i + batch_size < len(new_addresses):
            time.sleep(0.5)
    
    # Combine existing and new data
    all_user_data = []
    
    # First, add all existing data
    for addr in all_addresses:
        if addr.lower() in existing_data:
            all_user_data.append(existing_data[addr.lower()])
    
    # Then add new data
    all_user_data.extend(new_user_data)
    
    # Write all data to CSV
    csv_filename = 'addresses_fc.csv'
    fieldnames = ['address', 'fid', 'username', 'display_name', 'pfp_url', 'bio', 
                  'latitude', 'longitude', 'city', 'state', 'state_code', 
                  'country', 'country_code', 'follower_count', 'following_count', 
                  'verified_accounts', 'score']
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_user_data)
    
    # Print statistics
    new_users_found = sum(1 for user in new_user_data if user['fid'])
    total_users_found = sum(1 for user in all_user_data if user['fid'])
    
    print(f"\n=== Results ===")
    print(f"New addresses processed: {len(new_addresses)}")
    print(f"New Farcaster users found: {new_users_found}")
    print(f"Total Farcaster users in database: {total_users_found}")
    print(f"Total addresses in database: {len(all_user_data)}")
    print(f"Data saved to {csv_filename}")

if __name__ == "__main__":
    fetch_farcaster_users_incremental()