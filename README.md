# Farcaster Pro Analytics

Analytics dashboard and data pipeline for Farcaster Pro subscriptions ($120/year) on the Base blockchain.

## Overview

This project analyzes Farcaster Pro subscription transactions by:
1. Fetching USDC payment transactions from BaseScan API
2. Enriching blockchain addresses with Farcaster user profiles via Neynar API
3. Providing comprehensive analytics through an interactive Streamlit dashboard

## Features

- **Transaction Analysis**: Track ~10,000+ pro subscriptions totaling ~$1.25M in revenue
- **User Analytics**: Profile analysis including follower counts, Neynar scores, and FID distribution
- **Geographic Insights**: Global distribution of subscribers with country-level breakdowns
- **Interactive Dashboard**: Real-time filtering and exploration of subscriber data
- **Data Export**: Generate static HTML reports for sharing

## Setup

### Prerequisites

- Python 3.8+
- BaseScan API key
- Neynar API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pro_analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your API keys:
```env
BASESCAN_API_KEY=your_basescan_api_key
NEYNAR_API_KEY=your_neynar_api_key
```

## Usage

### 1. Fetch Transaction Data

Fetch ERC-20 token transactions from BaseScan:
```bash
python fetch_token_transactions.py
```

This creates:
- `erc20.json`: Raw transaction data
- `addresses.csv`: Unique sender addresses

### 2. Enrich with Farcaster Data

Map Ethereum addresses to Farcaster profiles:
```bash
python fetch_farcaster_users_incremental.py
```

This creates:
- `addresses_fc.csv`: Enriched user data with Farcaster profiles

### 3. Run Analytics Dashboard

Launch the interactive dashboard:
```bash
streamlit run dashboard.py
```

Or use the convenience script:
```bash
./run_dashboard.sh
```

Access the dashboard at http://localhost:8504

### 4. Generate Static Report

Create a shareable HTML report:
```bash
python generate_static_report.py
```

This creates:
- `analytics_report.html`: Standalone HTML file with embedded visualizations

## Data Schema

### addresses_fc.csv
- `address`: Ethereum address
- `fid`: Farcaster ID
- `username`: Farcaster username
- `display_name`: Display name
- `follower_count`: Number of followers
- `following_count`: Number following
- `score`: Neynar score (0-1)
- `latitude`, `longitude`: Geographic coordinates
- `city`, `state`, `country`: Location data

### Transaction Data
- **Contract**: USDC (0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913)
- **Recipient**: 0x0bdca19c9801bb484285362fd5dd0c94592c874c
- **Network**: Base blockchain
- **Price**: $120 USDC per subscription

## Analytics Insights

- **Total Subscribers**: ~10,380 (after filtering test transactions)
- **Total Revenue**: ~$1.25M
- **User Quality**: Mixed distribution of Neynar scores
- **Geographic Reach**: Global subscriber base
- **Peak Activity**: Most subscriptions within 6-8 hour window

## Project Structure

```
pro_analytics/
├── fetch_token_transactions.py      # BaseScan API integration
├── fetch_farcaster_users_incremental.py  # Neynar API integration
├── dashboard.py                     # Streamlit dashboard
├── generate_static_report.py        # Static report generator
├── run_dashboard.sh                 # Dashboard launcher
├── requirements.txt                 # Python dependencies
├── .env.example                     # Example environment variables
├── addresses.csv                    # Unique addresses from transactions
├── addresses_fc.csv                 # Enriched Farcaster user data
└── erc20.json                      # Raw transaction data
```

## License

MIT