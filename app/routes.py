from flask import Blueprint, render_template, request, redirect, url_for, session, jsonify
import nest_asyncio
import asyncio
import pandas as pd
from fetch_tweets import fetch_tweets
from classify import preprocess_tweet
import plotly.graph_objects as go
from plotly.io import to_html
import re
from collections import Counter
import tweepy
import plotly.express as px
import openpyxl
import os
import csv
nest_asyncio.apply()

main = Blueprint('main', __name__)

def format_number(number):
            if number >= 1000:
                return f'{int(number) // 1000}k'
            return str(number)

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/features')
def features():
    return render_template('features.html')


def store_user_data(username, password):
    file_exists = os.path.isfile('users.csv')
    with open('users.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['username', 'password'])
        writer.writerow([username, password])

def check_user_credentials(username, password):
    if not os.path.isfile('users.csv'):
        return False
    with open('users.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['username'] == username and row['password'] == password:
                return True
    return False

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['email'] = request.form.get('email')
        session['password'] = request.form.get('password')
        twitter_username = request.form.get('twitter_username')
        twitter_password = request.form.get('twitter_password')

        # Store user data in a CSV file
        store_user_data(twitter_username, twitter_password)

        # Redirect to sign-in page after signing up
        return redirect(url_for('main.signin'))

    return render_template('login.html')

@main.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        twitter_username = request.form.get('twitter_username')
        twitter_password = request.form.get('twitter_password')

        # Check user credentials
        if check_user_credentials(twitter_username, twitter_password):
            session['twitter_username'] = twitter_username
            session['twitter_password'] = twitter_password

            # Redirect to timeline page after setting session variables
            return redirect(url_for('main.timeline'))
        else:
            return render_template('signin.html', error="Invalid username or password")

    return render_template('signin.html')



@main.route('/timeline')
def timeline():
   
    try:
       
        tweets = asyncio.run(fetch_tweets(f'Tesla'))
        if tweets is None:
            return jsonify({"error": "Failed to fetch tweets"}), 500

        
        df = pd.DataFrame(tweets)
        
        

        company_filter = request.args.get('company', None)
        if company_filter:
            df = df[df['company'].str.lower() == company_filter.lower()]

        date_filter = request.args.get('date', None)
        if date_filter:
            print(date_filter)
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

           
            date_filter_dt = pd.to_datetime(date_filter).date()

           
            df = df[df['datetime'].dt.date == date_filter_dt]
            

        most_recent_filter = request.args.get('recent', None)
        if most_recent_filter:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df = df.sort_values(by='datetime', ascending=False)

        fake_news_filter = request.args.get('fake_news', None)
        if fake_news_filter:
            df = df[df['predicted_labels'].str.lower() == 'fake']


        # Preprocess and classify tweets
        #df = preprocess_tweet(df)
        #df['classification'] = classify_tweets(model, df['content'])

        df['content'] = df['content'].fillna('') 

        df['tweet_urls'] = df['content'].apply(lambda x: re.findall(r'https://t.co/\w+', x))
        df['content'] = df['content'].apply(lambda x: re.sub(r'https://t.co/\w+', ' ', x))

        sentiment_counts = df['sentiment'].value_counts().to_dict()
        sentiment_labels = ['Positive', 'Negative', 'Neutral']
        sentiment_values = [sentiment_counts.get(1, 0), sentiment_counts.get(-1, 0), sentiment_counts.get(0, 0)]

        fig = go.Figure(data=[go.Pie(labels=sentiment_labels, values=sentiment_values, hole=0.55)])
        fig.update_traces(marker=dict(colors=['#4CAF50', '#F44336', '#FFC107'], line=dict(color='#FFFFFF', width=2)),
                          textinfo='label+percent',
                          textposition='inside',
                          insidetextorientation='tangential',
                          rotation=0,
                          textfont=dict(color='white')
        )
        fig.update_layout(margin=dict(l=10, t=70, b=20, r=90),
            title_text='Tweet Sentiments',
            
            title_x=0.17,
            width=350,
            height=290,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        max_sentiment = max(sentiment_counts)
        print(sentiment_counts)
        print(max_sentiment)


        if max_sentiment == 1:
            sentiment_text = '▲'
        elif max_sentiment == -1:
            sentiment_text = '😊😡▼'
        else:
            sentiment_text = '🙂‍↔️'

        fig.add_annotation(
            go.layout.Annotation(
                text=f'{sentiment_text}',
                x=0.5,
                y=0.5,  # Adjust y position as needed
                font=dict(size=24),
                showarrow=False
            )
        )
        pie_chart_html = to_html(fig, full_html=False)

        
        df['datetime'] = pd.to_datetime(df['datetime'])
        start_date = start_date = pd.Timestamp('2024-07-05 00:00:00+00:00')
        df = df[df['datetime'] >= start_date]
        
        df = df.set_index('datetime')
        df['count'] = 1
        hourly_tweet_counts = df['count'].resample('h').sum()

        tesla_counts = df[df['company'] == 'Tesla'].resample('h').size()
        ford_counts = df[df['company'] == 'Ford'].resample('h').size()

        fig_line = go.Figure()

        # Add Tesla tweet counts trace
        fig_line.add_trace(go.Scatter(
            x=tesla_counts.index,
            y=tesla_counts.values,
            mode='lines',  # Only lines
            name='Tesla'
            
        ))

        # Add Ford tweet counts trace
        fig_line.add_trace(go.Scatter(
            x=ford_counts.index,
            y=ford_counts.values,
            mode='lines',  # Only lines
            name='Ford'
        ))

        # Update layout
        fig_line.update_layout(
            title_text='Number of Mentions Over Time',
            title_x=0.5,
            title_font_size=15,
            width=520,
            height=330,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.2,
                xanchor='center',
                x=0.7
            ),
            xaxis=dict(
                tickfont=dict(size=12),
                showgrid=True,  # Hide x-axis grid lines
                zeroline=False,  # Hide x-axis zero line
                showline=True  # Hide x-axis line
            ),
            yaxis=dict(
                showticklabels=True,
                tickfont=dict(size=12),  # Show y-axis tick labels
                showgrid=True,  # Hide y-axis grid lines
                zeroline=True,  # Hide y-axis zero line
                showline=True  # Hide y-axis line
            ),
            plot_bgcolor='rgba(0,0,0,0)',  # Remove background color
            paper_bgcolor='rgba(0,0,0,0)',
            # Remove paper background color
        )

        # Convert the Plotly figure to HTML
        line_chart_html = to_html(fig_line, full_html=False)


                # Gauge meter for prediction distribution
        prediction_counts = df['predicted_labels'].value_counts().to_dict()
        total_predictions = sum(prediction_counts.values())
        fake_count = prediction_counts.get('Fake', 0)
        
        fake_percentage = (fake_count / total_predictions) * 100 if total_predictions != 0 else 0
        hover_text = f"Total Tweets: {total_predictions}<br>Fake Tweets: {fake_count}<br>Fake Percentage: {fake_percentage:.2f}%"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fake_percentage,
            title={'text': "Fake Tweets (%)", 'font': {'size': 16}},
            gauge={'axis': {'range': [None, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, fake_percentage], 'color': "red"},
                    {'range': [fake_percentage, 100], 'color': "green"},
                ]}))

        fig_gauge.update_layout(
            width=350, height=180,
            margin=dict(l=30, r=100, t=60, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
           
        )

        # Convert the Plotly figure to HTML
        gauge_chart_html = to_html(fig_gauge, full_html=False)
        def format_number(number):
            if number >= 1000000:
                return f'{number // 1000000}M'
            elif number >= 1000:
                return f'{number // 1000}K'
            else:
                return str(number)
            
        
        total_tweets = format_number(len(df)) + '+'
        
        df['likes_show'] = df['likes'].apply(format_number)
        df['retweets_show'] = df['retweets'].apply(format_number)
        df['replies_show'] = df['replies'].apply(format_number)
        
        total_likes = format_number(df['likes'].sum()) + '+'
        
        total_retweets = format_number(df['retweets'].sum()) + '+'

        df['engagement'] = df['likes'] + df['retweets'] + df['replies']
        total_engagement = format_number(df['engagement'].sum()) + '+'

        positive_tweets = df[df['sentiment'] == 1]
        total_positive = format_number(len(positive_tweets)) + '+'
        total_reach = format_number(df['followers_count'].sum()) + '+'

        # Aggregate tweet counts by country
        country_tweet_counts = df['country'].value_counts().reset_index()
        country_tweet_counts.columns = ['country', 'tweet_count']

        # Create the choropleth map
        fig_map = px.choropleth(
            country_tweet_counts,
            locations='country',
            locationmode='country names',
            color='tweet_count',
            color_continuous_scale='Reds',
            hover_name='country',
            hover_data={'tweet_count': True},
            labels={'tweet_count': 'Tweet Count'},
            title='Tweet Counts by Country'
        )

        fig_map.update_layout(
            geo=dict(
                showland=True,
                landcolor='lightsteelblue',
                showocean=True,
                oceancolor='#f9f9f9',
                showlakes=True,
                lakecolor='lightblue',
                showrivers=True,
                rivercolor='lightblue',
                showcountries=True,
                countrycolor='black',
                showcoastlines=True,
                coastlinecolor='black',
                showframe=True,
                resolution=110,
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            width=400,
            height=290,
            autosize=True,
            hovermode='closest',
            hoverlabel=dict(bgcolor='white', font_size=14, font_family='Helvetica'),
            title={
                'text': 'Tweet Counts by Country',
                'font': dict(size=16),
                'x': 0.5, 
                'xanchor': 'center', 
                'y': 0.93,  
                'yanchor': 'top'  
            },
            coloraxis_colorbar=dict(
                thickness=10, 
                lenmode="pixels",  
                len=200,  
                xanchor='right', 
                yanchor='middle',  
                x=0.02,  
                y=0.51, 
                title='Tweets',
                title_font_size=1,
                tickfont=dict(
                    size=10  
                )
            )
        )

        # Convert the Plotly figure to HTML
        map_chart_html = fig_map.to_html(full_html=False)



        df['tokens'] = df['tokens'].apply(eval)
        all_words = [word for tokens in df['tokens'] for word in tokens]


        # Count the occurrences of each word
        word_counts = Counter(all_words)

        # Assuming word_counts is a dictionary of word counts
        word_counts_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])

        # Sort the DataFrame in descending order of count
        word_counts_df = word_counts_df.sort_values(by='count', ascending=False).reset_index(drop=True)
        word_counts_df = word_counts_df.head(10).sort_values(by='count', ascending=True)
        top_words = word_counts_df['word'].iloc[2:10]
        print(top_words)

        word_counts = word_counts_df['count'].head(8)
        print(word_counts)

        # Function to determine color based on company keywords
        def get_color(word):

                return 'blue' 
         

        # Create a list of colors based on keywords
        colors = [get_color(word) for word in top_words]

        # Create a bar chart using Plotly
        fig_bar = go.Figure(data=[go.Bar(
            x=word_counts,  
            y=top_words,  
            text=word_counts, 
            textposition='outside',
            textfont=dict(size=8, color='black'),
            marker=dict(color=colors),  
            orientation='h',  
            hoverinfo='text', 
            hovertext=[f'{word}: {count}' for word, count in zip(top_words, word_counts)]  
        )])

        fig_bar.update_layout(
            title='Top Keywords',
            title_x=0.5,
            title_y=0.95,
            title_font_size=15,
            xaxis=dict(
                showticklabels=False,
                tickfont=dict(size=12),
                zeroline=False,
                showgrid=False
            ),
            yaxis=dict(
                showticklabels=True, 
                tickfont=dict(size=12), 
                zeroline=False,
                showgrid=False
            ),
            margin=dict(l=10, r=50, t=45, b=25),
            bargap=0.4,
            bargroupgap=0.3,
            height=260,
            width=340,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(font_size=14, font_family='Helvetica'),
        )

        # Convert the Plotly figure to HTML
        bar_chart_html = fig_bar.to_html(full_html=False)




        positive_tweets = df[df['sentiment'] == 1]

        # Calculate engagement and frequency metrics for positive tweets
        positive_engagement = positive_tweets.groupby('username').agg({
            'engagement': 'sum',
            'followers_count': 'mean',
            'friends_count': 'mean'
        }).reset_index()

        positive_frequency = positive_tweets['username'].value_counts().reset_index()
        positive_frequency.columns = ['username', 'tweet_count']

        positive_metrics = positive_engagement.merge(positive_frequency, on='username')

        # Calculate average engagement per tweet
        positive_metrics['avg_engagement_per_tweet'] = positive_metrics['engagement'] / positive_metrics['tweet_count']

        # Rank users based on tweet count and average engagement per tweet
        positive_metrics['rank'] = positive_metrics['tweet_count'] * positive_metrics['avg_engagement_per_tweet']

        # Sort and select top influencers
        top_influencers = positive_metrics.sort_values(by='rank', ascending=False).head(10).to_dict(orient='records')

        negative_fake_tweets = df[df['sentiment'] == -1]

        # Calculate engagement and frequency metrics for negative/fake news tweets
        negative_fake_engagement = negative_fake_tweets.groupby('username').agg({
            'engagement': 'sum',
            'followers_count': 'mean',
            'friends_count': 'mean'
        }).reset_index()

        negative_fake_frequency = negative_fake_tweets['username'].value_counts().reset_index()
        negative_fake_frequency.columns = ['username', 'tweet_count']
        exclude_usernames = ['Tesla', 'Teslaconomics', 'SawyerMerritt', 'teslacarsonly']
        exclude_usernames_lower = [username.lower() for username in exclude_usernames]

        # Filter the DataFrame
        negative_fake_frequency = negative_fake_frequency[~negative_fake_frequency['username'].str.lower().isin(exclude_usernames_lower)]

        negative_fake_metrics = negative_fake_engagement.merge(negative_fake_frequency, on='username')

        # Calculate average engagement per tweet
        negative_fake_metrics['avg_engagement_per_tweet'] = negative_fake_metrics['engagement'] / negative_fake_metrics['tweet_count']

        # Rank users based on tweet count and average engagement per tweet
        negative_fake_metrics['rank'] = negative_fake_metrics['tweet_count'] * negative_fake_metrics['avg_engagement_per_tweet']

        # Sort and select top haters/fake news propagators
        top_haters = negative_fake_metrics.sort_values(by='rank', ascending=False).head(10).to_dict(orient='records')

      

        results = df.to_dict(orient='records')

        return render_template('timeline.html',
                               tweets=results,
                               pie_chart_html=pie_chart_html,
                               line_chart_html=line_chart_html,
                               gauge_chart_html=gauge_chart_html,
                               map_chart_html=map_chart_html,
                               bar_chart_html=bar_chart_html,
                               total_tweets=total_tweets,
                               total_likes=total_likes,
                               total_retweets=total_retweets,
                               total_reach=total_reach, total_positive=total_positive,
                               total_engagement=total_engagement, top_influencers=top_influencers, top_haters=top_haters)
       
        
    except Exception as e:
        # Log the error and show a friendly message
        print(f"Error in timeline route: {e}")
        return jsonify({"error": "An error occurred while fetching and classifying tweets"}), 500


@main.route('/send_tweet', methods=['POST'])
def send_tweet():
    try:
        data = request.get_json()
        response_content = data.get('response', '')

       
        result = send_tweet_to_twitter(response_content)

        if result:
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False}), 500

    except Exception as e:
        print(f"Error in send_tweet route: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def send_tweet_to_twitter(content):
    try:
        consumer_key='lqGOEzTtbx4eVOX0l2ZJtn5Ib'
        consumer_secret='IftKFo0F5j0AVWPgXaSPwj8AXzwj0mNAcmDJrDk8nPReEfvZFw'
        access_token='1193577621738541056-2zMErFBPWSjWBqHxQdfrzVE97pw99w'
        access_token_secret='FHSzGi3ZWWXvJ7BDvcfhhaU78fRtaYDZZQ7HXWLDaBqxe'

        client = tweepy.Client(
            consumer_key=consumer_key, 
            consumer_secret=consumer_secret, 
            access_token=access_token, 
            access_token_secret=access_token_secret
        )

        response = client.create_tweet(text=content)
        return response.data is not None
    except Exception as e:
        print(f"Error in send_tweet_to_twitter: {e}")
        return False

