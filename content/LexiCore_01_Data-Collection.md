---
Title: Collecting Data: Social Media and Media Attention (by Group "LexiCore")
Date: 2026-01-10 19:30
Category: Reflective Report
Tags: Group LexiCore
---

By Group "LexiCore"

# To Begin With
Our research topic focuses on the impact of media and social media Attention
on the excess returns of stocks of US technology companies. This involves two 
crucial data sources: social media text and traditional media news text. This
article will briefly describe the work we have done in this area, the problems
encountered during the process, and the methods we have attempted to solve 
them.

At the same time, in the article, we have presented the code used by some of 
the group members in their work. When you attempt to reproduce it, you should 
be aware that this is not the complete version. This is because we encountered 
some issues during the process of scrapping Reddit and X. If you also encounter 
similar problems and can identify some solutions or methods, please feel free 
to contact us. Thank you very much!

# Building a Structured News Dataset: A Technical Walkthrough

We systematically gathered news articles from two major sources—The Guardian and The New York Times—using their respective APIs. Here's a technical summary of our implementation, challenges faced, and how we structured the final dataset.

## Data Collection Architecture

We targeted a comprehensive date range from January 1, 2020, to December 31, 2025, covering pre-pandemic, pandemic, and post-pandemic periods including the ChatGPT launch era.

For **The Guardian**, we utilized the Open Platform API with daily iteration:

```python
# Date range generation
from datetime import datetime, timedelta

start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 12, 31)
dates = []

current_date = start_date
while current_date <= end_date:
    date = current_date.strftime('%Y-%m-%d')
    dates.append(str(date))
    current_date += timedelta(days=1)

# Section filtering and API calls
sections = ['better-business', 'business', 'business-to-business', 'money', 'science', 'technology', 'us-news']
news_contents = {}
news_title = {}

for days in dates:
    daily_content = []
    daily_title = []
    
    url = f"https://content.guardianapis.com/search?from-date={days}&to-date={days}&production-office=us&show-fields=all&page-size=100&lang=en&api-key=API_KEY"
    response = requests.get(url)
    data = response.json()
    
    for article in data['response']['results']:
        if article['sectionId'] not in sections:
            continue
            
        body = article['fields']['bodyText']
        title = article['webTitle']
        daily_content.append(body)
        daily_title.append(title)
    
    news_contents[f'{days}'] = daily_content
    news_title[f'{days}'] = daily_title
```

For **The New York Times**, we employed the Archive API with monthly batch processing:

```python
# Monthly data collection
from collections import defaultdict

sections = ['Business Day', 'Science', 'Technology', 'U.S.', 'Your Money']
result = defaultdict(list)

for y in range(0, 6):
    for m in range(1, 13):
        url = f"https://api.nytimes.com/svc/archive/v1/202{y}/{m}.json?api-key=API_KEY"
        response = requests.get(url)
        data = response.json()
        articles = data['response']['docs']
        
        for i, article in enumerate(articles):
            try:
                section_name = article.get('section_name', '')
                if section_name not in sections:
                    continue
                
                pub_date_str = article.get('pub_date', '')
                if pub_date_str:
                    date_key = pub_date_str[:10]
                else:
                    continue
                
                article_info = {
                    'abstract': article.get('abstract', ''),
                    'headline': article.get('headline', {}).get('main', ''),
                    'section_name': section_name,
                    'word_count': article.get('word_count', 0)
                }
                result[date_key].append(article_info)
            except Exception as e:
                print(f"Error processing article {i}: {e}")
                continue
```

## Technical Challenges & Solutions

**API Rate Limiting** presented significant hurdles. The Guardian's daily quota required segmenting our six-year timeline and implementing key rotation mid-process:

```python
# Example of key rotation strategy
api_keys = ['key1', 'key2', 'key3']
current_key_index = 0

def get_api_key():
    global current_key_index
    key = api_keys[current_key_index]
    current_key_index = (current_key_index + 1) % len(api_keys)
    return key
```

NYT's per-second limits necessitated implementing strategic pauses and processing data in smaller monthly batches:

```python
import time

# Adding delays between requests
time.sleep(1)  # Wait 1 second between requests
```

**Data Storage Evolution** revealed architectural insights. Our initial approach separated titles and content into distinct JSON files, which proved cumbersome for analysis. We subsequently developed a merge script:

```python
# Merging separate data files
with open('news_Guardian_title.json', 'r') as f:
    titles_data = json.load(f)

with open('news_Guardian_contents.json', 'r') as f:
    content_data = json.load(f)

merged_data = {}
for date in titles_data.keys():
    merged_data[date] = []
    titles = titles_data[date]
    contents = content_data[date]
    
    for i in range(len(titles)):
        article = {"title": titles[i], "content": contents[i]}
        merged_data[date].append(article)

# Save merged data
with open('news_Guardian_merged.json', 'w') as f:
    json.dump(merged_data, f, indent=2)
```

**Data Consistency** required careful normalization. We standardized date formats across sources, handled missing fields systematically, and implemented validation checks to ensure data integrity throughout the six-year collection period.

## Implementation & Workflow

Our Python implementation employed `requests` for API calls, `datetime` for date sequencing, and `json` for data persistence. The workflow modularized date generation, API communication, data parsing, and file operations:

```python
# Core workflow structure
def generate_dates(start_date, end_date):
    # Generate list of dates
    pass

def fetch_guardian_articles(date, api_key):
    # Fetch Guardian articles for a specific date
    pass

def fetch_nyt_articles(year, month, api_key):
    # Fetch NYT articles for a specific month
    pass

def save_to_json(data, filename):
    # Save data to JSON file
    pass

# Main execution flow
dates = generate_dates(start_date, end_date)
for date in dates:
    articles = fetch_guardian_articles(date, api_key)
    save_to_json(articles, f'guardian_{date}.json')
```

We added comprehensive logging and progress indicators to monitor collection status across extended runtimes:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track progress
processed_dates = 0
total_dates = len(dates)

for date in dates:
    logger.info(f"Processing date {date} ({processed_dates}/{total_dates})")
    # Process data
    processed_dates += 1
```

The final datasets are structured to support reproducible research, with The Guardian data organized as date-keyed article lists and NYT data including metadata on source, total dates, article counts, and generation timestamps. This foundation enables downstream text preprocessing, sentiment analysis, and quantitative modeling of media-market relationships.

This technical groundwork provides a clean, query-ready news corpus spanning 2,190 days, supporting empirical analysis of how business and technology media coverage intersects with financial market dynamics.

# Scraping Social Media: An Unsuccessful Attempt
We attempted to use the API methods introduced in the lecture note to scrape the social media posts from Reddit and X and categorize them to find the relevant information we needed. However, we encountered problems such as authorization issues, quotas, and query duration limitations.

## Try Scraping Reddit

We try to copy the following URL directly to the address bar of our browser and open it to test token:

https://api.pushshift.io/reddit/search/submission?size=1

OUTPUT: {"detail":"Not authenticated"}

Here, we encountered authorization issues. Then we turn to applying for Pushshift API.

OUTPUT: {"detail":"User is not an authorized moderator."}

Still, we do not have sufficient authority to carry out the above operations. Through the official contact method, we attempted to contact the Reddit administrators via email to inquire about obtaining API authorization, but were refused.

### Pushshift API Access Changes
By reviewing public sources, we found that around 2023, Pushshift tightened their control policies regarding the API.

**Before (≤ 2022)**

+ The Pushshift API was fully public
+ No authentication token was required
+ Widely used in academic research without access restrictions

**Now (post-2023)**

+ Pushshift has been taken back under Reddit’s authorization
+ API access is restricted
+ Most endpoints require authentication via a token
+ Tokens are not automatically granted and are typically tied to authorized Reddit moderator accounts.

As a result, unauthenticated requests now return access errors (e.g., “Not authenticated”), which prevents direct data collection without approved credentials.

## Scraping Twitter/X: Why You Can't Fetch Old Tweets
While trying to scarpe Twitter, we encountered a frustrating limitation with Twitter's API that many developers face: the inability to fetch tweets beyond 7 days using the standard search endpoint.

```python
import tweepy

client = tweepy.Client(bearer_token="token_b")

# This works for tweets from the last 7 days
response = client.search_recent_tweets(
    query="NVIDIA lang:en -is:retweet",
    start_time="2026-01-07T00:00:00Z",  # about 3 days ago
    end_time="2026-01-07T23:59:59Z",
    max_results=100
)

# This FAILS with 400 Bad Request
response = client.search_recent_tweets(
    query="NVIDIA lang:en -is:retweet",
    start_time="2026-01-01T00:00:00Z",  # More than 7 days ago
    end_time="2026-01-01T23:59:59Z",
    max_results=100
)
```
###Key constraints of Twitter API

After digging through Twitter's documentation, We confirmed the issue: Twitter's search_recent_tweets endpoint only provides access to tweets from the last 7 days. This is a hard limitation of their Essential access tier (the free tier).


**Recent Search (/2/tweets/search/recent):**

+ Maximum 7-day lookback window

+ Rate limits: 180 requests/15-minute window (per user)

+ 512 character limit for queries

**Full-Archive Search (/2/tweets/search/all):**

+ Access to entire Twitter archive

+ Only available with Academic Research or Enterprise access

+ Requires special approval and payment

Twitter API's 7-day limitation for recent search is a significant constraint for historical analysis projects. While workarounds exist, they require careful planning and continuous data collection rather than one-time historical pulls.

For researchers needing full historical access, applying for Twitter's Academic Research access through a university affiliation remains the only official solution.








