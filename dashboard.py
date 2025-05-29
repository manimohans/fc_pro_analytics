import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import numpy as np
from scipy import stats

# Set page config
st.set_page_config(
    page_title="Farcaster Pro Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Metrics text */
    [data-testid="metric-container"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Add spacing between sections */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Section spacing */
    .element-container {
        margin-bottom: 1.5rem !important;
    }
    
    /* Remove default Streamlit padding */
    .css-1d391kg {
        padding-top: 1rem !important;
    }
    
    /* Plotly charts spacing */
    .js-plotly-plot {
        margin-bottom: 3rem !important;
    }
    
    /* Minimalistic plot styling */
    .plotly .modebar {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

def apply_minimalist_theme(fig, title=None, height=600):
    """Apply consistent minimalist theme to all plots"""
    fig.update_layout(
        title=title,
        title_font=dict(size=22, family="Space Grotesk", color="#333"),
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=80, t=100, b=80),
        font=dict(family="Space Grotesk", size=12, color="#666"),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Space Grotesk"
        )
    )
    
    # Update all axes for minimalistic look
    fig.update_xaxes(
        gridcolor='rgba(200,200,200,0.2)', 
        zerolinecolor='rgba(200,200,200,0.3)',
        tickfont=dict(size=11, color="#666")
    )
    fig.update_yaxes(
        gridcolor='rgba(200,200,200,0.2)', 
        zerolinecolor='rgba(200,200,200,0.3)',
        tickfont=dict(size=11, color="#666")
    )
    
    return fig

@st.cache_data
def load_data():
    """Load and prepare data from CSV and JSON files"""
    
    # Load Farcaster user data (all addresses first)
    df_fc = pd.read_csv('addresses_fc.csv')
    
    # Load transaction data to get timestamps
    with open('erc20.json', 'r') as f:
        tx_data = json.load(f)
    
    # Create a mapping of address to timestamp
    address_to_timestamp = {}
    for tx in tx_data['result']:
        from_address = tx['from'].lower()
        timestamp = int(tx['timeStamp'])
        # Keep the earliest timestamp for each address
        if from_address not in address_to_timestamp or timestamp < address_to_timestamp[from_address]:
            address_to_timestamp[from_address] = timestamp
    
    # Add timestamp to dataframe
    df_fc['timestamp'] = df_fc['address'].str.lower().map(address_to_timestamp)
    df_fc['datetime'] = pd.to_datetime(df_fc['timestamp'], unit='s')
    
    # Convert to PST (UTC-8) 
    df_fc['datetime_pst'] = df_fc['datetime'] - pd.Timedelta(hours=7)  # Fixed: was -8, now -7
    
    # Sort by timestamp and remove first 7 test transactions
    df_fc = df_fc.sort_values('timestamp').iloc[8:].reset_index(drop=True)
    
    # Clean numeric columns
    df_fc['follower_count'] = pd.to_numeric(df_fc['follower_count'], errors='coerce')
    df_fc['following_count'] = pd.to_numeric(df_fc['following_count'], errors='coerce')
    df_fc['score'] = pd.to_numeric(df_fc['score'], errors='coerce')
    df_fc['fid'] = pd.to_numeric(df_fc['fid'], errors='coerce')
    
    # Remove rows without timestamps (addresses not in transaction data)
    df_fc = df_fc.dropna(subset=['timestamp'])
    
    # Print information about users with 0 followers to console
    zero_followers = df_fc[df_fc['follower_count'] == 0]
    if len(zero_followers) > 0:
        print(f"\n=== USERS WITH 0 FOLLOWERS ({len(zero_followers)} total) ===")
        for i, (idx, user) in enumerate(zero_followers.head(5).iterrows()):
            print(f"\n{i+1}. Address: {user['address']}")
            print(f"   FID: {user.get('fid', 'N/A')}")
            print(f"   Username: {user.get('username', 'N/A')}")
            print(f"   Display Name: {user.get('display_name', 'N/A')}")
            print(f"   Bio: {str(user.get('bio', 'N/A'))[:100]}{'...' if len(str(user.get('bio', ''))) > 100 else ''}")
            print(f"   Following: {user.get('following_count', 'N/A')}")
            print(f"   Score: {user.get('score', 'N/A')}")
            print(f"   Country: {user.get('country', 'N/A')}")
            if pd.notna(user.get('datetime_pst')):
                print(f"   Subscription Time: {user['datetime_pst']}")
        if len(zero_followers) > 5:
            print(f"\n   ... and {len(zero_followers) - 5} more users with 0 followers")
        print("=" * 50)
    
    # Print information about users with unknown follower count
    unknown_followers = df_fc[df_fc['follower_count'].isna()]
    if len(unknown_followers) > 0:
        print(f"\n=== USERS WITH UNKNOWN FOLLOWER COUNT ({len(unknown_followers)} total) ===")
        for i, (idx, user) in enumerate(unknown_followers.head(5).iterrows()):
            print(f"\n{i+1}. Address: {user['address']}")
            print(f"   FID: {user.get('fid', 'N/A')}")
            print(f"   Username: {user.get('username', 'N/A')}")
            print(f"   Display Name: {user.get('display_name', 'N/A')}")
            print(f"   Bio: {str(user.get('bio', 'N/A'))[:100]}{'...' if len(str(user.get('bio', ''))) > 100 else ''}")
            print(f"   Following Count: {user.get('following_count', 'N/A')}")
            print(f"   Score: {user.get('score', 'N/A')}")
            print(f"   Country: {user.get('country', 'N/A')}")
            print(f"   Raw follower_count value: {repr(user.get('follower_count', 'N/A'))}")
            if pd.notna(user.get('datetime_pst')):
                print(f"   Subscription Time: {user['datetime_pst']}")
        if len(unknown_followers) > 5:
            print(f"\n   ... and {len(unknown_followers) - 5} more users with unknown follower count")
        print("=" * 50)
    
    return df_fc

def create_followers_vs_time_plot(df):
    """Create follower count vs time plot"""
    df_sorted = df.sort_values('datetime_pst')
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Follower Count vs Purchase Time', 'Cumulative Transactions')
    )
    
    # Main scatter plot
    fig.add_trace(
        go.Scatter(
            x=df_sorted['datetime_pst'],
            y=df_sorted['follower_count'],
            mode='markers',
            marker=dict(
                size=6,
                color=df_sorted['score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score", x=1.02),
                opacity=0.7
            ),
            text=[f"User: {u}<br>Followers: {f:,}<br>Score: {s:.2f}" 
                  for u, f, s in zip(df_sorted['username'], 
                                     df_sorted['follower_count'], 
                                     df_sorted['score'])],
            hovertemplate='%{text}<br>Time: %{x}<extra></extra>',
            name='Users'
        ),
        row=1, col=1
    )
    
    # Add moving average
    window = 50
    if len(df_sorted) > window:
        df_sorted['follower_ma'] = df_sorted['follower_count'].rolling(window=window, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=df_sorted['datetime_pst'],
                y=df_sorted['follower_ma'],
                mode='lines',
                line=dict(color='red', width=3),
                name=f'{window}-user Moving Avg',
                hovertemplate='Avg Followers: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Cumulative transactions
    df_sorted['cumulative_count'] = range(1, len(df_sorted) + 1)
    fig.add_trace(
        go.Scatter(
            x=df_sorted['datetime_pst'],
            y=df_sorted['cumulative_count'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='lightblue'),
            name='Cumulative Users',
            hovertemplate='Total Users: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (PST)", row=2, col=1)
    fig.update_yaxes(title_text="Follower Count", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Total Users", row=2, col=1)
    
    # Apply minimalist theme
    apply_minimalist_theme(fig, "Follower Count vs Purchase Time Analysis", height=600)
    
    return fig

def create_score_correlations_plot(df):
    """Create score correlation plots"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Score vs Follower Count',
            'Score vs Following Count',
            'Score Distribution',
            'Score vs Follower/Following Ratio'
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12
    )
    
    # 1. Score vs Followers
    df_clean = df.dropna(subset=['score', 'follower_count'])
    df_clean = df_clean[df_clean['follower_count'] > 0]
    
    fig.add_trace(
        go.Scatter(
            x=df_clean['score'],
            y=df_clean['follower_count'],
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.5),
            text=df_clean['username'],
            hovertemplate='User: %{text}<br>Score: %{x:.2f}<br>Followers: %{y:,}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Score vs Following
    df_clean2 = df.dropna(subset=['score', 'following_count'])
    df_clean2 = df_clean2[df_clean2['following_count'] > 0]
    
    fig.add_trace(
        go.Scatter(
            x=df_clean2['score'],
            y=df_clean2['following_count'],
            mode='markers',
            marker=dict(size=4, color='green', opacity=0.5),
            text=df_clean2['username'],
            hovertemplate='User: %{text}<br>Score: %{x:.2f}<br>Following: %{y:,}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. Score Distribution
    scores_clean = df['score'].dropna()
    fig.add_trace(
        go.Histogram(
            x=scores_clean,
            nbinsx=30,
            marker_color='purple',
            opacity=0.7,
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 4. Score vs Follower/Following Ratio
    df_ratio = df.dropna(subset=['score', 'follower_count', 'following_count'])
    df_ratio = df_ratio[df_ratio['following_count'] > 0]
    df_ratio['ff_ratio'] = df_ratio['follower_count'] / df_ratio['following_count']
    df_ratio = df_ratio[df_ratio['ff_ratio'] < 50]
    
    if len(df_ratio) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_ratio['score'],
                y=df_ratio['ff_ratio'],
                mode='markers',
                marker=dict(size=4, color='orange', opacity=0.5),
                text=df_ratio['username'],
                hovertemplate='User: %{text}<br>Score: %{x:.2f}<br>F/F Ratio: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=4, col=1
        )
    
    # Update axes
    fig.update_xaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Score", row=2, col=1)
    fig.update_xaxes(title_text="Score", row=3, col=1)
    fig.update_xaxes(title_text="Score", row=4, col=1)
    
    fig.update_yaxes(title_text="Followers", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Following", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Follower/Following Ratio", row=4, col=1)
    
    # Calculate correlations
    corr_followers = df_clean['score'].corr(np.log10(df_clean['follower_count'] + 1))
    corr_following = df_clean2['score'].corr(np.log10(df_clean2['following_count'] + 1))
    
    # Apply minimalist theme
    title = f"Score Correlations (Followers: {corr_followers:.3f} | Following: {corr_following:.3f})"
    apply_minimalist_theme(fig, title, height=1000)
    fig.update_layout(showlegend=False)
    
    return fig

def create_score_distribution_plot(df):
    """Create detailed score distribution plot"""
    scores_clean = df['score'].dropna()
    
    # Create figure with single column layout
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'Score Distribution (Histogram)',
            'Score Percentiles',
            'Score Quality Segments',
            'Average Score Over Time'
        ],
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12
    )
    
    # 1. Histogram
    fig.add_trace(
        go.Histogram(
            x=scores_clean,
            nbinsx=50,
            marker_color='rgba(55, 126, 184, 0.7)',
            marker_line=dict(color='rgba(55, 126, 184, 1)', width=1),
            name='User Count',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Box plot for percentiles
    fig.add_trace(
        go.Box(
            y=scores_clean,
            boxpoints='outliers',
            marker_color='rgba(255, 127, 14, 0.7)',
            line_color='rgba(255, 127, 14, 1)',
            name='Score Distribution',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. Score segments
    score_ranges = [
        ('Low (0.0-0.3)', (scores_clean >= 0.0) & (scores_clean < 0.3)),
        ('Medium (0.3-0.6)', (scores_clean >= 0.3) & (scores_clean < 0.6)),
        ('High (0.6-0.8)', (scores_clean >= 0.6) & (scores_clean < 0.8)),
        ('Very High (0.8-1.0)', (scores_clean >= 0.8) & (scores_clean <= 1.0))
    ]
    
    segment_counts = [scores_clean[mask].count() for _, mask in score_ranges]
    segment_labels = [label for label, _ in score_ranges]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    fig.add_trace(
        go.Bar(
            x=segment_labels,
            y=segment_counts,
            marker_color=colors,
            text=[f'{count:,}<br>({count/len(scores_clean)*100:.1f}%)' for count in segment_counts],
            textposition='auto',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 4. Average score over time
    df_time = df.dropna(subset=['score', 'datetime_pst']).copy()
    df_time = df_time.sort_values('datetime_pst')
    
    # Group by time intervals (every 5 minutes) and calculate average score
    df_time['time_group'] = df_time['datetime_pst'].dt.floor('5min')  # 5-minute intervals
    time_scores = df_time.groupby('time_group').agg({
        'score': ['mean', 'count'],
        'datetime_pst': 'first'
    }).reset_index()
    
    # Flatten column names
    time_scores.columns = ['time_group', 'avg_score', 'user_count', 'datetime_pst']
    
    # Only include time periods with at least 3 users for better averaging
    time_scores = time_scores[time_scores['user_count'] >= 3]
    
    if len(time_scores) > 0:
        fig.add_trace(
            go.Scatter(
                x=time_scores['time_group'],
                y=time_scores['avg_score'],
                mode='lines+markers',
                line=dict(color='rgba(75, 0, 130, 1)', width=2),
                marker=dict(size=6, color='rgba(75, 0, 130, 0.8)'),
                text=[f'Time: {t}<br>Avg Score: {s:.3f}<br>Users: {c}' 
                      for t, s, c in zip(time_scores['time_group'], 
                                        time_scores['avg_score'], 
                                        time_scores['user_count'])],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ),
            row=4, col=1
        )
        
        # Add trend line
        if len(time_scores) > 5:
            x_numeric = np.arange(len(time_scores))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, time_scores['avg_score'])
            trend_line = slope * x_numeric + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=time_scores['time_group'],
                    y=trend_line,
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'Trend (R²={r_value**2:.3f})',
                    showlegend=False
                ),
                row=4, col=1
            )
    
    # Update axes
    fig.update_xaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="Score Segments", row=3, col=1)
    fig.update_xaxes(title_text="Time (PST)", row=4, col=1)
    
    fig.update_yaxes(title_text="User Count", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="User Count", row=3, col=1)
    fig.update_yaxes(title_text="Average Score", row=4, col=1)
    
    # Calculate key statistics
    mean_score = scores_clean.mean()
    median_score = scores_clean.median()
    std_score = scores_clean.std()
    high_quality_users = (scores_clean >= 0.8).sum()
    
    # Apply minimalist theme
    title = f"Score Distribution Analysis (μ={mean_score:.2f}, σ={std_score:.2f}, median={median_score:.2f})<br><sub>High Quality Users (≥0.8): {high_quality_users:,} ({high_quality_users/len(scores_clean)*100:.1f}%)</sub>"
    apply_minimalist_theme(fig, title, height=1000)
    fig.update_layout(showlegend=False)
    
    return fig

def create_follower_distribution_plot(df):
    """Create follower count distribution with focus on low-follower users"""
    df_clean = df.dropna(subset=['follower_count']).copy()
    df_clean = df_clean[df_clean['follower_count'] >= 0]  # Remove negative values
    
    # Create single column layout
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'All Users - Follower Distribution (Log Scale)',
            'Users with < 10 Followers (Detail View)', 
            'Follower Count Segments',
            'Low-Follower User Analysis'
        ],
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12
    )
    
    # 1. Overall distribution (log scale for better visibility)
    # Create bins and get usernames for hover text
    hist, bin_edges = np.histogram(df_clean['follower_count'], bins=50)
    
    # Create hover text with random usernames for each bin
    hover_texts = []
    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        users_in_bin = df_clean[(df_clean['follower_count'] >= bin_start) & 
                                (df_clean['follower_count'] < bin_end)]
        
        if len(users_in_bin) > 0:
            # Get up to 25 random usernames from this bin
            sample_size = min(25, len(users_in_bin))
            sample_users = users_in_bin.sample(n=sample_size)['username'].tolist()
            
            hover_text = f"Range: {bin_start:.0f}-{bin_end:.0f}<br>"
            hover_text += f"Count: {hist[i]}<br>"
            hover_text += f"Sample users:<br>"
            hover_text += "<br>".join([f"@{u}" for u in sample_users[:25]])
        else:
            hover_text = f"Range: {bin_start:.0f}-{bin_end:.0f}<br>Count: 0"
        
        hover_texts.append(hover_text)
    
    fig.add_trace(
        go.Histogram(
            x=df_clean['follower_count'],
            nbinsx=50,
            marker_color='rgba(31, 119, 180, 0.7)',
            marker_line=dict(color='rgba(31, 119, 180, 1)', width=1),
            name='All Users',
            showlegend=False,
            hovertemplate='%{text}<extra></extra>',
            text=hover_texts
        ),
        row=1, col=1
    )
    
    # 2. Users with < 10 followers (detailed view)
    low_followers_df = df_clean[df_clean['follower_count'] < 10]
    if len(low_followers_df) > 0:
        # Create hover text for each follower count (0-9)
        hover_texts_low = []
        hist_low, bin_edges_low = np.histogram(low_followers_df['follower_count'], bins=10, range=(0, 10))
        
        for i in range(len(hist_low)):
            exact_count = i
            users_with_count = low_followers_df[low_followers_df['follower_count'] == exact_count]
            
            if len(users_with_count) > 0:
                sample_size = min(25, len(users_with_count))
                sample_users = users_with_count.sample(n=sample_size)['username'].tolist()
                
                hover_text = f"Followers: {exact_count}<br>"
                hover_text += f"Count: {len(users_with_count)}<br>"
                hover_text += f"Sample users:<br>"
                hover_text += "<br>".join([f"@{u}" for u in sample_users[:25]])
            else:
                hover_text = f"Followers: {exact_count}<br>Count: 0"
            
            hover_texts_low.append(hover_text)
        
        fig.add_trace(
            go.Histogram(
                x=low_followers_df['follower_count'],
                nbinsx=10,  # One bin per follower count (0-9)
                marker_color='rgba(255, 127, 14, 0.7)',
                marker_line=dict(color='rgba(255, 127, 14, 1)', width=1),
                name='< 10 Followers',
                showlegend=False,
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts_low
            ),
            row=2, col=1
        )
    
    # 3. Follower count segments
    follower_ranges = [
        ('0 followers', df_clean['follower_count'] == 0),
        ('1-9 followers', (df_clean['follower_count'] >= 1) & (df_clean['follower_count'] <= 9)),
        ('10-99 followers', (df_clean['follower_count'] >= 10) & (df_clean['follower_count'] <= 99)),
        ('100-999 followers', (df_clean['follower_count'] >= 100) & (df_clean['follower_count'] <= 999)),
        ('1K-9.9K followers', (df_clean['follower_count'] >= 1000) & (df_clean['follower_count'] <= 9999)),
        ('10K+ followers', df_clean['follower_count'] >= 10000)
    ]
    
    segment_counts = [df_clean[mask].shape[0] for _, mask in follower_ranges]
    segment_labels = [label for label, _ in follower_ranges]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
    
    # Create hover text with usernames for each segment
    segment_hover_texts = []
    for label, mask in follower_ranges:
        users_in_segment = df_clean[mask]
        count = len(users_in_segment)
        
        if count > 0:
            sample_size = min(25, count)
            sample_users = users_in_segment.sample(n=sample_size)['username'].tolist()
            
            hover_text = f"{label}<br>"
            hover_text += f"Count: {count:,} ({count/len(df_clean)*100:.1f}%)<br>"
            hover_text += f"Sample users:<br>"
            hover_text += "<br>".join([f"@{u}" for u in sample_users[:25]])
        else:
            hover_text = f"{label}<br>Count: 0"
        
        segment_hover_texts.append(hover_text)
    
    fig.add_trace(
        go.Bar(
            x=segment_labels,
            y=segment_counts,
            marker_color=colors,
            text=[f'{count:,}<br>({count/len(df_clean)*100:.1f}%)' for count in segment_counts],
            textposition='auto',
            showlegend=False,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=segment_hover_texts
        ),
        row=3, col=1
    )
    
    # 4. Low-follower user analysis - show exact follower counts for < 10
    if len(low_followers_df) > 0:
        exact_counts = low_followers_df['follower_count'].value_counts().sort_index()
        
        # Create hover text with usernames for each exact count
        exact_hover_texts = []
        for follower_count in exact_counts.index:
            users_with_exact_count = low_followers_df[low_followers_df['follower_count'] == follower_count]
            count = len(users_with_exact_count)
            
            if count > 0:
                sample_size = min(25, count)
                sample_users = users_with_exact_count.sample(n=sample_size)['username'].tolist()
                
                hover_text = f"{follower_count} followers<br>"
                hover_text += f"Count: {count} users<br>"
                hover_text += f"Sample users:<br>"
                hover_text += "<br>".join([f"@{u}" for u in sample_users[:25]])
            else:
                hover_text = f"{follower_count} followers<br>Count: 0"
            
            exact_hover_texts.append(hover_text)
        
        fig.add_trace(
            go.Bar(
                x=exact_counts.index,
                y=exact_counts.values,
                marker_color='rgba(214, 39, 40, 0.7)',
                marker_line=dict(color='rgba(214, 39, 40, 1)', width=1),
                text=[f'{count} users' for count in exact_counts.values],
                textposition='auto',
                showlegend=False,
                hovertemplate='%{customdata}<extra></extra>',
                customdata=exact_hover_texts
            ),
            row=4, col=1
        )
    
    # Update axes
    fig.update_xaxes(title_text="Follower Count", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Follower Count", row=2, col=1)
    fig.update_xaxes(title_text="Follower Segments", row=3, col=1)
    fig.update_xaxes(title_text="Exact Follower Count", row=4, col=1)
    
    fig.update_yaxes(title_text="User Count", row=1, col=1)
    fig.update_yaxes(title_text="User Count", row=2, col=1)
    fig.update_yaxes(title_text="User Count", row=3, col=1)
    fig.update_yaxes(title_text="User Count", row=4, col=1)
    
    # Calculate key statistics
    total_users = len(df_clean)
    zero_followers = (df_clean['follower_count'] == 0).sum()
    low_followers_count = (df_clean['follower_count'] < 10).sum()
    median_followers = df_clean['follower_count'].median()
    mean_followers = df_clean['follower_count'].mean()
    
    # Apply minimalist theme
    title = f"Follower Distribution Analysis (Total: {total_users:,} users)<br><sub>Low-follower users (<10): {low_followers_count:,} ({low_followers_count/total_users*100:.1f}%) | Zero followers: {zero_followers:,} ({zero_followers/total_users*100:.1f}%) | Median: {median_followers:.0f} | Mean: {mean_followers:.0f}</sub>"
    apply_minimalist_theme(fig, title, height=1000)
    fig.update_layout(showlegend=False)
    
    return fig

def create_fid_distribution_plot(df):
    """Create FID distribution analysis with score correlations"""
    df_clean = df.dropna(subset=['fid', 'score']).copy()
    df_clean['fid'] = pd.to_numeric(df_clean['fid'], errors='coerce')
    df_clean = df_clean.dropna(subset=['fid'])
    
    # Create single column layout
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'FID Distribution (Log Scale)',
            'FID vs Score Correlation',
            'FID vs Score Over Time',
            'FID Segments Analysis'
        ],
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12
    )
    
    # 1. FID Distribution histogram
    fig.add_trace(
        go.Histogram(
            x=df_clean['fid'],
            nbinsx=50,
            marker_color='rgba(148, 103, 189, 0.7)',
            marker_line=dict(color='rgba(148, 103, 189, 1)', width=1),
            name='FID Distribution',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. FID vs Score scatter plot
    fig.add_trace(
        go.Scatter(
            x=df_clean['fid'],
            y=df_clean['score'],
            mode='markers',
            marker=dict(
                size=4,
                color=df_clean['follower_count'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Followers", x=1.1)
            ),
            text=[f"@{u}<br>FID: {f:,}<br>Score: {s:.2f}<br>Followers: {fc:,}" 
                  for u, f, s, fc in zip(df_clean['username'], df_clean['fid'], 
                                         df_clean['score'], df_clean['follower_count'])],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. FID vs Score over time (animated by time)
    if 'datetime_pst' in df_clean.columns:
        df_time = df_clean.dropna(subset=['datetime_pst']).sort_values('datetime_pst')
        
        # Group by time intervals to show progression
        df_time['time_group'] = df_time['datetime_pst'].dt.floor('30min')  # 30-minute intervals
        
        # Calculate average score for each FID range over time
        fid_ranges = [
            (0, 100000, 'Early adopters (0-100K)'),
            (100000, 500000, 'Mid adopters (100K-500K)'),
            (500000, 1000000, 'Late adopters (500K-1M)'),
            (1000000, float('inf'), 'Recent users (1M+)')
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (fid_min, fid_max, label) in enumerate(fid_ranges):
            range_data = df_time[(df_time['fid'] >= fid_min) & (df_time['fid'] < fid_max)]
            if len(range_data) > 0:
                time_avg = range_data.groupby('time_group').agg({
                    'score': 'mean',
                    'fid': 'count'
                }).reset_index()
                time_avg = time_avg[time_avg['fid'] >= 3]  # At least 3 users per time period
                
                fig.add_trace(
                    go.Scatter(
                        x=time_avg['time_group'],
                        y=time_avg['score'],
                        mode='lines+markers',
                        name=label,
                        line=dict(color=colors[i], width=2),
                        marker=dict(size=6),
                        showlegend=True
                    ),
                    row=3, col=1
                )
    
    # 4. FID Segments analysis
    fid_segments = [
        ('0-10K (OG)', (df_clean['fid'] >= 0) & (df_clean['fid'] < 10000)),
        ('10K-100K (Early)', (df_clean['fid'] >= 10000) & (df_clean['fid'] < 100000)),
        ('100K-500K (Mid)', (df_clean['fid'] >= 100000) & (df_clean['fid'] < 500000)),
        ('500K-1M (Late)', (df_clean['fid'] >= 500000) & (df_clean['fid'] < 1000000)),
        ('1M+ (Recent)', df_clean['fid'] >= 1000000)
    ]
    
    segment_stats = []
    for label, mask in fid_segments:
        segment_data = df_clean[mask]
        if len(segment_data) > 0:
            segment_stats.append({
                'segment': label,
                'count': len(segment_data),
                'avg_score': segment_data['score'].mean(),
                'avg_followers': segment_data['follower_count'].mean()
            })
    
    if segment_stats:
        segment_df = pd.DataFrame(segment_stats)
        
        # Create grouped bar chart
        fig.add_trace(
            go.Bar(
                x=segment_df['segment'],
                y=segment_df['avg_score'],
                name='Avg Score',
                marker_color='lightblue',
                yaxis='y',
                offsetgroup=1,
                text=[f"{s:.2f}" for s in segment_df['avg_score']],
                textposition='auto'
            ),
            row=4, col=1
        )
        
        # Add count as text annotation
        for i, row in segment_df.iterrows():
            fig.add_annotation(
                x=row['segment'],
                y=row['avg_score'] + 0.02,
                text=f"{row['count']} users",
                showarrow=False,
                row=4, col=1
            )
    
    # Update axes
    fig.update_xaxes(title_text="FID", type="log", row=1, col=1)
    fig.update_xaxes(title_text="FID", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Time (PST)", row=3, col=1)
    fig.update_xaxes(title_text="FID Segments", row=4, col=1)
    
    fig.update_yaxes(title_text="User Count", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="Average Score", row=3, col=1)
    fig.update_yaxes(title_text="Average Score", row=4, col=1)
    
    # Calculate key statistics
    median_fid = df_clean['fid'].median()
    mean_fid = df_clean['fid'].mean()
    
    # Correlation between FID and score
    correlation = df_clean['fid'].corr(df_clean['score'])
    
    # Apply minimalist theme
    title = f"FID Distribution Analysis (Median: {median_fid:,.0f} | Mean: {mean_fid:,.0f})<br><sub>FID-Score Correlation: {correlation:.3f} | Lower FID = Earlier User</sub>"
    apply_minimalist_theme(fig, title, height=1000)
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0.02, y=0.3, bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.1)')
    )
    
    return fig

def create_unknown_followers_timeline(df):
    """Create timeline analysis for users with unknown follower counts"""
    # Get users with unknown follower counts
    unknown_users = df[df['follower_count'].isna()].copy()
    
    if len(unknown_users) == 0:
        st.info("No users with unknown follower counts found.")
        return None
    
    # Sort by timestamp
    unknown_users = unknown_users.sort_values('datetime_pst')
    
    # Create subplots with single column layout
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            'Unknown Followers Timeline',
            'Unknown Users by Hour of Day',
            'Score Distribution of Unknown Users',
            'FID Distribution of Unknown Users'
        ],
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12
    )
    
    # 1. Timeline scatter plot
    fig.add_trace(
        go.Scatter(
            x=unknown_users['datetime_pst'],
            y=list(range(len(unknown_users))),  # Convert range to list
            mode='markers',
            marker=dict(
                size=8,
                color=unknown_users['score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score", x=1.02, len=0.35, y=0.825)
            ),
            text=[f"@{u}<br>FID: {f}<br>Score: {s:.2f}<br>Following: {fc}" 
                  for u, f, s, fc in zip(unknown_users['username'], 
                                         unknown_users['fid'], 
                                         unknown_users['score'],
                                         unknown_users['following_count'])],
            hovertemplate='%{text}<br>Time: %{x}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Hour of day distribution
    unknown_users['hour'] = unknown_users['datetime_pst'].dt.hour
    hour_counts = unknown_users['hour'].value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=hour_counts.index,
            y=hour_counts.values,
            marker_color='rgba(255, 127, 14, 0.7)',
            text=hour_counts.values,
            textposition='auto',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. Score distribution of unknown users
    scores_unknown = unknown_users['score'].dropna()
    if len(scores_unknown) > 0:
        fig.add_trace(
            go.Histogram(
                x=scores_unknown,
                nbinsx=20,
                marker_color='rgba(44, 160, 44, 0.7)',
                marker_line=dict(color='rgba(44, 160, 44, 1)', width=1),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # 4. FID distribution of unknown users
    fids_unknown = unknown_users['fid'].dropna()
    if len(fids_unknown) > 0:
        fig.add_trace(
            go.Histogram(
                x=fids_unknown,
                nbinsx=20,
                marker_color='rgba(148, 103, 189, 0.7)',
                marker_line=dict(color='rgba(148, 103, 189, 1)', width=1),
                showlegend=False
            ),
            row=4, col=1
        )
    
    # Update axes
    fig.update_xaxes(title_text="Time (PST)", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day (PST)", row=2, col=1)
    fig.update_xaxes(title_text="Score", row=3, col=1)
    fig.update_xaxes(title_text="FID", type="log", row=4, col=1)
    
    fig.update_yaxes(title_text="User Index", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=4, col=1)
    
    # Apply minimalist theme
    title = f"Unknown Followers Analysis ({len(unknown_users)} users)<br><sub>These users have missing follower count data from the Neynar API</sub>"
    apply_minimalist_theme(fig, title, height=900)
    
    # Add summary statistics
    st.markdown(f"""
    **Unknown Users Statistics:**
    - Total: {len(unknown_users)} users ({len(unknown_users)/len(df)*100:.1f}% of all users)
    - Average Score: {unknown_users['score'].mean():.3f}
    - Median FID: {unknown_users['fid'].median():,.0f}
    - Time Range: {unknown_users['datetime_pst'].min()} to {unknown_users['datetime_pst'].max()}
    """)
    
    return fig

def create_high_fid_analysis(df):
    """Create detailed analysis for FIDs above 1M split by 10K segments"""
    # Filter for FIDs above 1M
    df_high_fid = df[df['fid'] >= 1_000_000].copy()
    
    if len(df_high_fid) == 0:
        st.info("No users with FID above 1M found.")
        return None
    
    # Create 10K segments
    df_high_fid['fid_segment'] = ((df_high_fid['fid'] - 1_000_000) // 10_000) * 10_000 + 1_000_000
    
    # Create figure with single column layout
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'Distribution of FIDs Above 1M (10K Segments)',
            'Average Score by FID Segment',
            'Follower Distribution by FID Segment'
        ],
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.12
    )
    
    # 1. Count by segment
    segment_counts = df_high_fid.groupby('fid_segment').size().reset_index(name='count')
    segment_counts['segment_label'] = segment_counts['fid_segment'].apply(
        lambda x: f"{x/1000:.0f}K-{(x+10000)/1000:.0f}K"
    )
    
    fig.add_trace(
        go.Bar(
            x=segment_counts['segment_label'],
            y=segment_counts['count'],
            marker_color='rgba(75, 0, 130, 0.7)',
            text=segment_counts['count'],
            textposition='auto',
            hovertemplate='%{x}<br>Count: %{y}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Average score by segment
    segment_scores = df_high_fid.groupby('fid_segment').agg({
        'score': 'mean',
        'fid': 'count'
    }).reset_index()
    segment_scores['segment_label'] = segment_scores['fid_segment'].apply(
        lambda x: f"{x/1000:.0f}K-{(x+10000)/1000:.0f}K"
    )
    
    fig.add_trace(
        go.Scatter(
            x=segment_scores['segment_label'],
            y=segment_scores['score'],
            mode='lines+markers',
            marker=dict(size=10, color='rgba(255, 127, 14, 0.8)'),
            line=dict(color='rgba(255, 127, 14, 0.8)', width=3),
            text=[f"Avg Score: {s:.3f}<br>Users: {c}" 
                  for s, c in zip(segment_scores['score'], segment_scores['fid'])],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. Follower distribution by segment
    follower_stats = df_high_fid.groupby('fid_segment').agg({
        'follower_count': ['mean', 'median', 'std']
    }).round(0)
    follower_stats.columns = ['avg_followers', 'median_followers', 'std_followers']
    follower_stats = follower_stats.reset_index()
    follower_stats['segment_label'] = follower_stats['fid_segment'].apply(
        lambda x: f"{x/1000:.0f}K-{(x+10000)/1000:.0f}K"
    )
    
    # Create grouped bar chart
    fig.add_trace(
        go.Bar(
            x=follower_stats['segment_label'],
            y=follower_stats['avg_followers'],
            name='Average',
            marker_color='rgba(31, 119, 180, 0.7)',
            offsetgroup=1,
            text=[f"{v:,.0f}" for v in follower_stats['avg_followers']],
            textposition='auto',
            showlegend=True
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=follower_stats['segment_label'],
            y=follower_stats['median_followers'],
            name='Median',
            marker_color='rgba(44, 160, 44, 0.7)',
            offsetgroup=2,
            text=[f"{v:,.0f}" for v in follower_stats['median_followers']],
            textposition='auto',
            showlegend=True
        ),
        row=3, col=1
    )
    
    # Update axes
    fig.update_xaxes(title_text="FID Segment", row=1, col=1)
    fig.update_xaxes(title_text="FID Segment", row=2, col=1)
    fig.update_xaxes(title_text="FID Segment", row=3, col=1)
    
    fig.update_yaxes(title_text="User Count", row=1, col=1)
    fig.update_yaxes(title_text="Average Score", row=2, col=1)
    fig.update_yaxes(title_text="Follower Count", row=3, col=1)
    
    # Apply minimalist theme
    total_high_fid = len(df_high_fid)
    avg_score = df_high_fid['score'].mean()
    avg_followers = df_high_fid['follower_count'].mean()
    
    title = f"High FID Analysis (1M+): {total_high_fid:,} users<br><sub>Average Score: {avg_score:.3f} | Average Followers: {avg_followers:,.0f}</sub>"
    apply_minimalist_theme(fig, title, height=900)
    
    # Update legend position
    fig.update_layout(
        legend=dict(x=0.02, y=0.15, bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.1)')
    )
    
    # Additional statistics
    st.markdown(f"""
    **High FID (1M+) Statistics:**
    - Total Users: {total_high_fid:,}
    - FID Range: {df_high_fid['fid'].min():,.0f} to {df_high_fid['fid'].max():,.0f}
    - Average Score: {avg_score:.3f}
    - Average Followers: {avg_followers:,.0f}
    - Total Segments: {len(segment_counts)}
    """)
    
    return fig

def create_recent_transactions_analysis(df):
    """Analyze the last 1500 transactions"""
    # Get the last 1500 transactions
    df_recent = df.sort_values('datetime_pst').tail(1500).copy()
    
    # Calculate rolling averages for smoother visualization
    df_recent['rolling_avg_followers'] = df_recent['follower_count'].rolling(window=50, min_periods=1).mean()
    df_recent['rolling_avg_score'] = df_recent['score'].rolling(window=50, min_periods=1).mean()
    df_recent['transaction_number'] = range(1, len(df_recent) + 1)
    
    # Calculate statistics for summary
    avg_followers = df_recent['follower_count'].mean()
    median_followers = df_recent['follower_count'].median()
    avg_score = df_recent['score'].mean()
    time_start = df_recent['datetime_pst'].min().strftime('%Y-%m-%d %H:%M')
    time_end = df_recent['datetime_pst'].max().strftime('%Y-%m-%d %H:%M')
    
    # Display summary info in a centered box
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h4 style="margin: 0;">Time Range: {time_start} to {time_end} PST</h4>
        <p style="margin: 10px 0 0 0; font-size: 16px;">
            <strong>Avg Followers:</strong> {avg_followers:,.0f} | 
            <strong>Median:</strong> {median_followers:,.0f} | 
            <strong>Avg Score:</strong> {avg_score:.3f}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create subplots with more spacing
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'Follower Count Distribution - Percentile Bands (Log Scale)',
            'Score Distribution Over Last 1500 Transactions',
            'FID Distribution Over Last 1500 Transactions (Focus on Higher FIDs)'
        ],
        row_heights=[0.25, 0.25, 0.50],
        vertical_spacing=0.15
    )
    
    # 1. Follower count distribution over time - using log scale and percentiles
    # Calculate percentiles for better visualization
    df_recent['follower_percentile'] = df_recent['follower_count'].rank(pct=True) * 100
    
    # Create rolling percentiles
    window_size = 50
    df_recent['rolling_p25'] = df_recent['follower_count'].rolling(window=window_size, min_periods=1).quantile(0.25)
    df_recent['rolling_p50'] = df_recent['follower_count'].rolling(window=window_size, min_periods=1).quantile(0.50)
    df_recent['rolling_p75'] = df_recent['follower_count'].rolling(window=window_size, min_periods=1).quantile(0.75)
    df_recent['rolling_p90'] = df_recent['follower_count'].rolling(window=window_size, min_periods=1).quantile(0.90)
    
    # Add percentile bands
    fig.add_trace(
        go.Scatter(
            x=df_recent['datetime_pst'],
            y=df_recent['rolling_p90'],
            mode='lines',
            name='90th percentile',
            line=dict(color='rgba(31, 119, 180, 0.3)', width=1),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_recent['datetime_pst'],
            y=df_recent['rolling_p75'],
            mode='lines',
            name='75th percentile',
            line=dict(color='rgba(31, 119, 180, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_recent['datetime_pst'],
            y=df_recent['rolling_p50'],
            mode='lines',
            name='Median (50th)',
            line=dict(color='#1f77b4', width=3),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_recent['datetime_pst'],
            y=df_recent['rolling_p25'],
            mode='lines',
            name='25th percentile',
            line=dict(color='rgba(31, 119, 180, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add scatter plot with size based on follower count
    fig.add_trace(
        go.Scatter(
            x=df_recent['datetime_pst'],
            y=df_recent['follower_count'],
            mode='markers',
            name='Individual users',
            marker=dict(
                size=np.log10(df_recent['follower_count'] + 1) * 3,  # Log scale for size
                color=df_recent['score'],
                colorscale='Viridis',
                opacity=0.4,
                line=dict(width=0)
            ),
            text=[f"User: {row['username']}<br>Followers: {row['follower_count']:,}<br>Score: {row['score']:.3f}" 
                  for _, row in df_recent.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 2. Score spread - box plot over time windows
    # Create 15 bins of 100 transactions each
    df_recent['bin'] = pd.cut(df_recent['transaction_number'], bins=15, labels=False)
    
    for bin_num in range(15):
        bin_data = df_recent[df_recent['bin'] == bin_num]
        if len(bin_data) > 0:
            # Calculate date range for this bin
            bin_start = bin_data['datetime_pst'].min().strftime('%m/%d %H:%M')
            bin_end = bin_data['datetime_pst'].max().strftime('%m/%d %H:%M')
            fig.add_trace(
                go.Box(
                    y=bin_data['score'],
                    name=f'{bin_start}<br>to<br>{bin_end}',
                    marker_color='#2ca02c',
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # 3. FID spread - with focus on higher FIDs
    # Separate data by FID ranges
    df_below_100k = df_recent[df_recent['fid'] <= 100_000]
    df_100k_1m = df_recent[(df_recent['fid'] > 100_000) & (df_recent['fid'] < 1_000_000)]
    df_above_1m = df_recent[df_recent['fid'] >= 1_000_000]
    
    # Add a single point at y=0 to represent all FIDs <= 100K
    if len(df_below_100k) > 0:
        # Create a summary point for all <= 100K FIDs
        fig.add_trace(
            go.Scatter(
                x=[df_recent['datetime_pst'].min(), df_recent['datetime_pst'].max()],
                y=[0, 0],
                mode='lines',
                name=f'FID ≤ 100K ({len(df_below_100k)} users)',
                line=dict(color='lightgray', width=3),
                showlegend=True
            ),
            row=3, col=1
        )
    
    # FIDs 100K-1M (detailed spread)
    if len(df_100k_1m) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_100k_1m['datetime_pst'],
                y=df_100k_1m['fid'],
                mode='markers',
                name='FID 100K-1M',
                marker=dict(
                    color=df_100k_1m['score'],
                    colorscale='Blues',
                    size=6,
                    opacity=0.7,
                    colorbar=dict(title="Score<br>(100K-1M)", x=1.02, len=0.3, y=0.35)
                ),
                text=[f"User: {row['username']}<br>Time: {row['datetime_pst'].strftime('%Y-%m-%d %H:%M:%S')}<br>FID: {row['fid']:,}<br>Score: {row['score']:.3f}" 
                      for _, row in df_100k_1m.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ),
            row=3, col=1
        )
    
    # FIDs >= 1M (detailed spread)
    if len(df_above_1m) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_above_1m['datetime_pst'],
                y=df_above_1m['fid'],
                mode='markers',
                name='FID ≥ 1M',
                marker=dict(
                    color=df_above_1m['score'],
                    colorscale='Reds',
                    size=8,
                    opacity=0.8,
                    colorbar=dict(title="Score<br>(≥1M)", x=1.12, len=0.3, y=0.35)
                ),
                text=[f"User: {row['username']}<br>Time: {row['datetime_pst'].strftime('%Y-%m-%d %H:%M:%S')}<br>FID: {row['fid']:,}<br>Score: {row['score']:.3f}" 
                      for _, row in df_above_1m.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ),
            row=3, col=1
        )
    
    # Add trend lines for high FID groups
    if len(df_100k_1m) > 5:
        x_numeric = (df_100k_1m['datetime_pst'] - df_100k_1m['datetime_pst'].min()).dt.total_seconds()
        z = np.polyfit(x_numeric, df_100k_1m['fid'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=df_100k_1m['datetime_pst'],
                y=p(x_numeric),
                mode='lines',
                name='Trend 100K-1M',
                line=dict(color='blue', dash='dash', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
    
    if len(df_above_1m) > 5:
        x_numeric = (df_above_1m['datetime_pst'] - df_above_1m['datetime_pst'].min()).dt.total_seconds()
        z = np.polyfit(x_numeric, df_above_1m['fid'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=df_above_1m['datetime_pst'],
                y=p(x_numeric),
                mode='lines',
                name='Trend ≥1M',
                line=dict(color='red', dash='dash', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Add horizontal reference line at 1M only
    fig.add_hline(y=1_000_000, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
    
    # Add annotation for 1M line
    fig.add_annotation(
        x=df_recent['datetime_pst'].min(),
        y=1_000_000,
        text="1M",
        showarrow=False,
        yshift=10,
        row=3, col=1
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date/Time (PST)", row=1, col=1)
    fig.update_xaxes(title_text="Time Bins", row=2, col=1)
    fig.update_xaxes(title_text="Date/Time (PST)", row=3, col=1)
    
    fig.update_yaxes(title_text="Follower Count (Log Scale)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="FID", row=3, col=1)
    
    # Set y-axis to start from 0 and show up to max FID
    max_fid = df_recent['fid'].max()
    fig.update_yaxes(range=[-50000, max_fid * 1.05], row=3, col=1)
    
    title = f"Last 1500 Transactions Analysis"
    apply_minimalist_theme(fig, title, height=1300)
    
    # Update legend position for better visibility
    fig.update_layout(
        legend=dict(
            x=1.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )
    
    # Add additional statistics about timing
    total_duration = df_recent['datetime_pst'].max() - df_recent['datetime_pst'].min()
    hours = total_duration.total_seconds() / 3600
    transactions_per_hour = 1500 / hours if hours > 0 else 0
    
    # Calculate FID distribution
    fids_above_1_09m = len(df_recent[df_recent['fid'] > 1_090_000])
    fids_below_1_09m = len(df_recent[df_recent['fid'] <= 1_090_000])
    percent_above = (fids_above_1_09m / 1500) * 100
    percent_below = (fids_below_1_09m / 1500) * 100
    
    # Additional FID breakdown for the focused ranges
    fids_below_100k = len(df_recent[df_recent['fid'] <= 100_000])
    fids_100k_1m = len(df_recent[(df_recent['fid'] > 100_000) & (df_recent['fid'] < 1_000_000)])
    fids_above_1m = len(df_recent[df_recent['fid'] >= 1_000_000])
    
    # Calculate follower distribution for the last 1200
    followers_0 = len(df_recent[df_recent['follower_count'] == 0])
    followers_1_10 = len(df_recent[(df_recent['follower_count'] >= 1) & (df_recent['follower_count'] <= 10)])
    followers_100_1k = len(df_recent[(df_recent['follower_count'] >= 100) & (df_recent['follower_count'] < 1000)])
    followers_1k_10k = len(df_recent[(df_recent['follower_count'] >= 1000) & (df_recent['follower_count'] < 10000)])
    followers_10k_plus = len(df_recent[df_recent['follower_count'] >= 10000])
    followers_unknown = len(df_recent[df_recent['follower_count'].isna()])
    
    st.markdown(f"""
    **Transaction Timing Statistics:**
    - Total Duration: {hours:.1f} hours ({total_duration.days} days)
    - Average Rate: {transactions_per_hour:.1f} transactions/hour
    - Peak Hour: {df_recent.groupby(df_recent['datetime_pst'].dt.hour).size().idxmax()}:00 PST
    - Most Active Day: {df_recent['datetime_pst'].dt.date.value_counts().index[0]}
    
    **FID Distribution (Account Age):**
    - FIDs > 1.09M (Newer accounts): **{fids_above_1_09m:,}** ({percent_above:.1f}%)
    - FIDs ≤ 1.09M (Older accounts): **{fids_below_1_09m:,}** ({percent_below:.1f}%)
    
    **FID Range Breakdown:**
    - FIDs ≤ 100K: **{fids_below_100k}** ({fids_below_100k/15:.1f}%)
    - FIDs 100K-1M: **{fids_100k_1m}** ({fids_100k_1m/15:.1f}%)
    - FIDs ≥ 1M: **{fids_above_1m}** ({fids_above_1m/15:.1f}%)
    
    **Follower Distribution (Last 1500):**
    - 0 followers: **{followers_0}** ({followers_0/15:.1f}%)
    - 1-10 followers: **{followers_1_10}** ({followers_1_10/15:.1f}%)
    - 100-1K followers: **{followers_100_1k}** ({followers_100_1k/15:.1f}%)
    - 1K-10K followers: **{followers_1k_10k}** ({followers_1k_10k/15:.1f}%)
    - 10K+ followers: **{followers_10k_plus}** ({followers_10k_plus/15:.1f}%)
    - Unknown: **{followers_unknown}** ({followers_unknown/15:.1f}%)
    """)
    
    return fig

def create_geography_plot(df):
    """Create geographic distribution plot"""
    has_lat_long = df['latitude'].notna() & df['longitude'].notna()
    has_country = df['country_code'].notna()
    
    if has_lat_long.sum() == 0:
        st.warning("No geographic data available for mapping")
        return None
    
    # World map
    df_geo = df[has_lat_long].copy()
    
    fig = px.scatter_geo(
        df_geo,
        lat='latitude',
        lon='longitude',
        color='score',
        size='follower_count',
        hover_data=['username', 'city', 'country'],
        color_continuous_scale='Viridis'
    )
    
    # Apply minimalist theme
    apply_minimalist_theme(fig, 'Global Distribution of Pro Subscribers', height=500)
    
    return fig

def main():
    # Title and header
    st.title("📊 Farcaster Pro Analytics Dashboard")
    st.markdown("**Analysis of Pro Subscribers ($120/year)**")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Key Metrics Section
    st.markdown("### 📊 Key Metrics")
    
    time_range = df['datetime_pst'].max() - df['datetime_pst'].min()
    hours = time_range.total_seconds() / 3600
    
    # Prepare metrics data
    metrics_data = [
        ("Total Subscribers", f"{len(df):,}", "#1f77b4"),
        ("Total Revenue", f"${len(df) * 120:,}", "#2ca02c"),
        ("Time Span", f"{hours:.1f} hours", "#ff7f0e"),
        ("Users per Hour", f"{len(df) / hours:.0f}", "#d62728"),
        ("Average Score", f"{df['score'].mean():.3f}", "#9467bd"),
        ("High Score Users (≥0.9)", f"{len(df[df['score'] >= 0.9]):,}", "#17becf")
    ]
    
    # Create columns for metrics
    cols = st.columns(6)
    
    for col, (label, value, color) in zip(cols, metrics_data):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color: {color}15;
                    border: 2px solid {color};
                    border-radius: 10px;
                    padding: 20px 10px;
                    text-align: center;
                    height: 100px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 12px; color: #666; margin-bottom: 5px;">
                        {label}
                    </div>
                    <div style="font-size: 20px; font-weight: bold; color: {color};">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Follower Distribution Breakdown
    st.markdown("### 📊 Follower Distribution Breakdown")
    
    # Create follower ranges
    follower_ranges = [
        ("0 followers", df['follower_count'] == 0),
        ("1-10", (df['follower_count'] >= 1) & (df['follower_count'] <= 10)),
        ("10-100", (df['follower_count'] > 10) & (df['follower_count'] <= 100)),
        ("100-1K", (df['follower_count'] > 100) & (df['follower_count'] <= 1000)),
        ("1K-10K", (df['follower_count'] > 1000) & (df['follower_count'] <= 10000)),
        ("10K-100K", (df['follower_count'] > 10000) & (df['follower_count'] <= 100000)),
        ("100K+", df['follower_count'] > 100000),
        ("Unknown", df['follower_count'].isna())
    ]
    
    # Calculate counts and percentages
    follower_data = []
    for label, mask in follower_ranges:
        count = len(df[mask])
        percentage = (count / len(df)) * 100
        follower_data.append((label, count, percentage))
    
    # Create columns for the distribution
    cols = st.columns(8)
    
    # Define colors for each range (added gray for unknown)
    colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#1f77b4', '#9467bd', '#17becf', '#7f7f7f']
    
    for i, (col, (label, count, percentage), color) in enumerate(zip(cols, follower_data, colors)):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color: {color}20;
                    border: 2px solid {color};
                    border-radius: 10px;
                    padding: 15px;
                    text-align: center;
                    height: 120px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 14px; font-weight: bold; color: {color};">
                        {label}
                    </div>
                    <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">
                        {count:,}
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        {percentage:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Add spacing after metrics
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Plots with better spacing
    st.header("🕐 1. Follower Count vs Purchase Time")
    fig1 = create_followers_vs_time_plot(df)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.header("📈 2. Neynar Score Correlations")
    fig2 = create_score_correlations_plot(df)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.header("📊 3. Score Distribution Analysis")
    fig3 = create_score_distribution_plot(df)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.header("👥 4. Follower Count Distribution")
    fig4 = create_follower_distribution_plot(df)
    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.header("🆔 5. FID Distribution Analysis")
    fig5 = create_fid_distribution_plot(df)
    st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.header("❓ 6. Unknown Followers Analysis")
    fig6 = create_unknown_followers_timeline(df)
    if fig6:
        st.plotly_chart(fig6, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.header("🚀 7. High FID Analysis (1M+)")
    fig7 = create_high_fid_analysis(df)
    if fig7:
        st.plotly_chart(fig7, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.header("📈 8. Recent Transactions Analysis (Last 1500)")
    fig8 = create_recent_transactions_analysis(df)
    st.plotly_chart(fig8, use_container_width=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.header("🌍 9. Geographic Distribution")
    
    # Geographic data completeness
    total_users = len(df)
    has_lat_long = df['latitude'].notna() & df['longitude'].notna()
    has_country = df['country_code'].notna()
    has_city = df['city'].notna()
    
    st.info(f"""
    **Geographic Data Completeness:**
    - Users with lat/long: {has_lat_long.sum():,} ({has_lat_long.sum()/total_users*100:.1f}%)
    - Users with country: {has_country.sum():,} ({has_country.sum()/total_users*100:.1f}%)
    - Users with city: {has_city.sum():,} ({has_city.sum()/total_users*100:.1f}%)
    """)
    
    if has_lat_long.sum() > 0:
        fig9 = create_geography_plot(df)
        if fig9:
            st.plotly_chart(fig9, use_container_width=True)
        
        # Country breakdown
        if has_country.sum() > 0:
            st.subheader("Top Countries")
            country_stats = df[has_country].groupby('country').agg({
                'follower_count': ['count', 'sum', 'mean'],
                'score': 'mean'
            }).round(2)
            
            country_stats.columns = ['user_count', 'total_followers', 'avg_followers', 'avg_score']
            country_stats = country_stats.sort_values('user_count', ascending=False).head(10)
            
            st.dataframe(country_stats, use_container_width=True)
    else:
        st.warning("No geographic coordinates available for mapping")
    
    # Sidebar with filters
    st.sidebar.header("📊 Data Filters")
    
    # Score filter
    min_score = st.sidebar.slider("Minimum Score", 0.0, 1.0, 0.0, 0.1)
    
    # Follower filter
    min_followers = st.sidebar.number_input("Minimum Followers", 0, int(df['follower_count'].max()), 0)
    
    # Apply filters
    filtered_df = df[
        (df['score'] >= min_score) & 
        (df['follower_count'] >= min_followers)
    ]
    
    if len(filtered_df) != len(df):
        st.sidebar.write(f"Filtered: {len(filtered_df):,} / {len(df):,} users")
        
        # Show filtered summary
        st.sidebar.subheader("Filtered Stats")
        st.sidebar.write(f"Avg Score: {filtered_df['score'].mean():.3f}")
        st.sidebar.write(f"Avg Followers: {filtered_df['follower_count'].mean():,.0f}")
        st.sidebar.write(f"Total Reach: {filtered_df['follower_count'].sum()/1e6:.1f}M")

if __name__ == "__main__":
    main()