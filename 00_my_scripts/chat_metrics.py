from flask import Flask, render_template_string
import plotly.io as pio
import json
import pandas as pd
from collections import Counter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# Load JSON logs from file
def load_logs(log_file = 'chat_log.json'):
    with open(log_file, 'r') as f:
        logs = [json.loads(line.strip()) for line in f.readlines()]

    # Convert logs to DataFrame for easier analysis
    df = pd.DataFrame(logs)

     # Exclude rows with a nan user_ip
    df = df.dropna(subset=['user_ip'])

    # Create a dictionary that maps each unique IP address to a unique number
    ip_to_number = {ip: number for number, ip in enumerate(df['user_ip'].unique())}
    print(ip_to_number)

    # Replace user_ip values with unique numbers
    df['user_ip'] = df['user_ip'].map(ip_to_number)

    # Specify the exact format of the timestamp to improve parsing
    timestamp_format = '%Y-%m-%d %H:%M:%S,%f'
    df['timestamp'] = pd.to_datetime(df['timestamp'], format=timestamp_format)

    return df

# Function to get user activity by counting the number of messages per user IP
def get_user_activity(df):
    # Filter for chat events
    chat_df = df[df['event'] == 'chat']

    user_activity = chat_df['user_ip'].value_counts().reset_index()
    user_activity.columns = ['user_ip', 'message_count']
    return user_activity

# Function to analyze user feedback (like/dislike)
def analyze_feedback(df):
    feedback = df[df['event'] == 'feedback']
    liked_disliked = feedback['liked'].value_counts().reset_index()
    liked_disliked.columns = ['feedback', 'count']
    return liked_disliked

# Function to count tokens used in each chat
def count_tokens(df):
    tokens = df[df['event'] == 'chat']['token_used']
    tokens = tokens[tokens != '?'].astype(int)  # Convert to integers
    return tokens

# Function to get the number of unique users per day
def unique_users_per_day(df):
    daily_unique_users = df.groupby(df['timestamp'].dt.date)['user_ip'].nunique()
    return daily_unique_users.reset_index(name='unique_users')


def main():
    print("Running dashboard script...")
    
    # Load the chat logs
    df = load_logs()

    # Aggregate metrics for summary table
    total_logged_messages = len(df)
    total_users = df['user_ip'].nunique()
    total_tokens = count_tokens(df).sum()
    total_questions = df[df['event'] == 'chat'].shape[0]
    total_received_feedback = df[df['event'] == 'feedback'].shape[0]


    # Prepare data for all metrics
    user_activity = get_user_activity(df)
    feedback_df = analyze_feedback(df)
    tokens = count_tokens(df)
    unique_users_df = unique_users_per_day(df)

    # Create a subplot figure with a grid of 3x2 (3 rows, 2 columns)
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{'type': 'table', 'colspan': 2}, None],
            [{'type': 'xy'}, {'type': 'domain'}],
            [{'type': 'xy'}, {'type': 'xy'}]
        ],
        subplot_titles=(
            "Summary Metrics",
            "User Activity (Messages per User IP)",
            "User Feedback Distribution",
            "Token Usage Distribution",
            "Unique Users Per Day"
        )
    )

    # Add the summary table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                align='left',
                fill_color='#007BFF',  # Change to your preferred color (header background color)
                font=dict(color='white', size=14)  # Change to your preferred text color and size
            ),
            cells=dict(
                values=[
                    ["Total logged Messages", "Total Different Users", "Total Tokens", "Total Questions", "Total Received Feedback"],
                    [total_logged_messages, total_users, total_tokens, total_questions, total_received_feedback]
                ],
                align='left',
                fill_color=[['#f0f8ff', '#e6f2ff'] * 2],  # Alternating row colors
                font=dict(color='black', size=12)  # Change to your preferred text color and size
            )
        ),
        row=1, col=1
    )

    # Plot 1: User Activity
    fig.add_trace(
        go.Bar(x=user_activity['user_ip'], y=user_activity['message_count']),
        row=2, col=1
    )

    # Plot 2: Feedback Analysis
    fig.add_trace(
        go.Pie(labels=feedback_df['feedback'], values=feedback_df['count']),
        row=2, col=2
    )

    # Plot 3: Token Usage Distribution
    fig.add_trace(
        go.Histogram(x=tokens, nbinsx=20),
        row=3, col=1
    )

    # Plot 4: Unique Users Per Day
    fig.add_trace(
        go.Scatter(x=unique_users_df['timestamp'], y=unique_users_df['unique_users'], mode='lines+markers'),
        row=3, col=2
    )

    # Update the layout for better appearance
    fig.update_layout(
        height=1000, width=1200,
        title_text="Chatbot Metrics Dashboard",
        showlegend=True
    )

    # Display the entire dashboard
    # fig.show()
    return fig

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    fig = main()
    div = pio.to_html(fig, full_html=False)
    return render_template_string("""<html><body>{{ div|safe }}</body></html>""", div=div)

if __name__ == '__main__':
    app.run(port=5000)  # specify the port number here