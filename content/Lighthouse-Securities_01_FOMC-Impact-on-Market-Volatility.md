---
Title: FOMC Impact on Market Volatility (by Group "Lighthouse Securities")
Date: 2026-01-10 12:00
Category: Reflective Report
Tags: Group Lighthouse Securities
---

By Group "Lighthouse Securities"

> **Lighthouse Securities**:
>The trusted light for your financial journey

## 1. Introduction
Lighthouse Securities has been closely monitoring the impact of Federal Open Market Committee (FOMC) meeting minutes after the past years. We often see that specific wording or sentiment in these announcements can lead to significant market reactions across global markets.

The relationships between FOMC meeting minutes and interest rates are well-documented, with changes in monetary policy often leading to shifts in investor sentiment and market volatility. However, the impact on broader market volatility is a complex phenomenon influenced by various factors. We aim to shed light on these dynamics by analyzing historical data surrounding FOMC minutes and relating them to market volatility measures.

The results of our analysis will be published seperately, with these blog posts focused on our personal reflections on the project as we iterate towards completion.


## 2. Ideation
Before starting any project, we need to brainstorm to gather potential ideas. Our goal is to find a topic that combines natural-language processing (NLP) and finance. During this process, we found that most ideas have a **trade-off between uniqueness and feasibility**.

Some ideas are frequently found online and have existing research literature:

- Sentiment analysis of financial news articles and their impact on stock prices
- Predicting stock market trends using social media sentiment analysis
- Analyzing earnings call transcripts to predict stock price movements
- Financial analysts view on a stock and its correlation with stock performance

Other ideas are unique but may face challenges in data availability or complexity:

- CEO social media activity and its influence on stock performance
    - *Limited access to social media data, not all CEOs are active on (the same) social media*
- Industries mentioned in China's Five-Year Plans and their stock market performance
    - *Requires extensive data collection and analysis of government documents*

And some ideas strike a balance between uniqueness and feasibility:

- Analyzing FOMC meeting minutes and their impact on market volatility
- Mentions of AI and their implementation in earnings call transcripts and their effect on stock prices

After careful consideration, we decided to focus on analyzing FOMC meeting minutes and their impact on market volatility. We will aim to add unique insights be offering a cross-country comparison between the FED and ECB meeting minutes, and their respective market impacts.

## 3. Background research

Research and literature on using Natural Language Processing (NLP) decoding the hidden meaning behind the central bank communications is quite popular. This literature makes use of advanced machine learning models like FinBERT and XLNet, allowing for the quantification of the impact of FOMC statements on financial markets.

### 3.1 Key methodologies
1. **Cosine similarity**: A measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. In text analysis, it is often used to compare the similarity between two documents or pieces of text by representing them as vectors in a multi-dimensional space.

2. **FinBERT**: A pre-trained language model specifically designed for financial text analysis. It is based on the BERT architecture and fine-tuned on a large corpus of financial documents, enabling it to capture the nuances of financial language effectively.
    - For example, Denisa Marc (California State University, Fullerton) employed the FinBERT to perform an overall sentiment classification on all FOMC statement texts, automatically labeling them as "positive", "negative", or "neutral". This analysis revealed the distribution and evolving trends in the tone of policy statements across different historical periods, such as the Financial Crisis and the COVID-19 pandemic.
3. **XLNet**: An advanced language model that improves upon BERT by using a generalized autoregressive pretraining method, which allows it to better understand the context and semantics of financial texts.
    - For example, Amy Handlan's (Brown University) approach: First, predict expectation shifts using "alternative" FOMC documents, which describe the economy but do not set policy. Then subtract the average expectation shift from these documents from the expectation shifts predicted using the actual FOMC statements to isolate the policy-induced component.
4. **LLMs**: Pre-trained LLMs, such as provided by [gtfintechlab](https://github.com/gtfintechlab/WorldCentralBanks) are fine-tuned based on data from many central banks across the world, allowing for a more comprehensive understanding of central bank communications.
    - They focus on providing annotations for stance (hawkish, dovish, neutral), temporal focus (current, forward-looking), and uncertainty (certain, uncertain).

While each methodology has its strengths and weaknesses, they are within a scope for our project to implement. We aim to explore these methodologies further in our analysis and see how they can be applied to our dataset of FOMC meeting minutes and market volatility measures. Depending on the outcomes, we can see what techniques fits our project best.

## 4. Data workflow

The data workflow consists of 10 steps:

1. Data collection
2. Data structure
3. Data storage
4. Data cleaning
5. Data validation
6. Merging
7. Data transformation
8. Data analysis
9. Community engagement and public relations
10. Quality control

The blogposts will feature mainly on steps 1 to 7, and itself acts as step 9. The data analysis and results are part of the seperately published analysis presentation.

### 4.1 Data collection

As the first step in the data workflow, we created a list of data we need:

1. FED FOMC meeting minutes
2. ECB monetary policy meeting minutes
3. US stock market data
4. EU stock market data

#### 4.1.1 FED FOMC meeting minutes
To scrape FOMC meeting minutes, we first found two useful APIs on [Fed website](https://www.federalreserve.gov/monetarypolicy/materials/), which return all the links we need, and we downloaded them as [`final-hist.json`](https://www.federalreserve.gov/monetarypolicy/materials/assets/final-hist.json) and [`final-recent.json`](https://www.federalreserve.gov/monetarypolicy/materials/assets/final-recent.json). Then we extracted all the FOMC minutes related URLs and used Python's `requests` library to fetch the HTML content of the Federal Reserve's website, and then used `BeautifulSoup` to parse the HTML and extract the relevant text data. Below is a sample code snippet that demonstrates how to achieve this:

```python
import os
import json


def get_minutes_urls():
    with open(r'./data/final-hist.json', 'r', encoding='utf-8') as f:
        hist = json.load(f)

    with open(r'./data/final-recent.json', 'r', encoding='utf-8') as f:
        recent = json.load(f)

    data = {
        'hist': hist,
        'recent': recent,
    }

    count = 0
    res = {}

    for key in data:
        for item in data[key]['mtgitems']:
            if item['type'] == 'Mn':
                for file in item['files']:
                    try:
                        if 'fomc' in file['url'] and (
                                ('minutes' in file['url'] or 'MINUTES' in file['url']) or file['name'] == 'HTML'):
                            count += 1
                            print(count, file['url'])
                            url = 'https://www.federalreserve.gov' + file['url'] if file['url'].startswith('/') else \
                                file['url']
                            res[item['d']] = url
                            break
                    except KeyError:
                        count += 1
                        print(count, file)

    with open(r'./data/fomc_minutes_urls.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

    print(f'Total minutes found: {len(res)}')

    return res
```
```python
import requests
from bs4 import BeautifulSoup

def get_minutes_text(url):
    # get <p> text from url
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in paragraphs])
    return text
```

#### 4.1.2 ECB monetary policy meeting minutes
To collect the ECB meeting minutes, we faced a different challenge: the ECB's meeting accounts page relies on dynamic content that is not easily accessible via static HTTP requests. To solve this, we used **Selenium** with a headless Chrome browser to simulate a real user and wait for the JavaScript to render the meeting list. Once the list was loaded, we filtered for URLs containing the `/press/accounts/` pattern. For each identified URL, we then utilized `BeautifulSoup` to parse and extract the meeting text from the relevant HTML elements.

```python
from selenium import webdriver
from bs4 import BeautifulSoup

def get_all_meeting_urls_selenium():
    driver = setup_driver()
    driver.get("https://www.ecb.europa.eu/press/accounts/html/index.en.html")
    
    # Wait for dynamic content to load
    time.sleep(5) 
    
    all_links = driver.find_elements(By.TAG_NAME, "a")
    for link in all_links:
        href = link.get_attribute('href')
        if href and '/press/accounts/' in href and 'mg' in href:
            # Collect and save unique meeting URLs
            ...
```

```python
def download_meeting(url, output_dir, meeting_title):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract titles, dates, and text content
    content_area = soup.find('div', class_='ecb-langContent') or soup.find('main')
    paragraphs = content_area.find_all(['p', 'h2', 'h3', 'h4'])
    full_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs])
    ...
```

#### 4.1.3 US stock market data
To capture the historical performance and volatility of the US market, we utilized the `yfinance` library to retrieve daily closing prices for several key benchmarks starting from 1993. Our selection includes the S&P 500 (`^GSPC`) for large-cap stocks, the S&P 400 (`^SP400`) for mid-cap, and the Russell 2000 (`^RUT`) for small-cap equities. Additionally, we gathered data for the VIX (`^VIX`) to serve as our primary measure of US market fear and volatility.

```python
import yfinance as yf

# Indices: S&P 500, S&P 400 MidCap, Russell 2000, VIX
tickers = ['^GSPC', '^SP400', '^RUT', '^VIX']
data = yf.download(tickers, start='1993-01-01', progress=False)

# Extract and rename closing prices
close_prices = data['Close'].dropna()
close_prices.columns = ['us_large', 'us_mid', 'us_small', 'vix']
```

#### 4.1.4 EU stock market data
For the European market, we adopted a similar approach by gathering ETF data that represents different market capitalization segments. We focused on the Vanguard FTSE Europe ETF (`VGK`) for large-cap, iShares MSCI Europe Small-Cap ETF (`IEUS`) for mid-cap exposure, and the WisdomTree Europe SmallCap Dividend Fund (`DFE`) for small-cap stocks.

Regarding volatility, our measure of choice is the EURO STOXX 50 Volatility Index (VSTOXX). The ETF data was retrieved via `yfinance`, while the VSTOXX index measures the market's expectation of 30-day volatility of the EURO STOXX 50 index, derived from the prices of EURO STOXX 50 options.

The VSTOXX data was directly downloaded from the [STOXX website](https://www.stoxx.com/data-index-details?symbol=V2TX). This is one of the observations we made during the process: US data is more readily available through APIs like `yfinance`, while European data often requires accessing specific financial websites or databases.

```python
# Market cap-weighted European ETF portfolio
etf_portfolio = {
    'Large_Cap_ETF': 'VGK',
    'Mid_Cap_ETF': 'IEUS',
    'Small_Cap_ETF': 'DFE',
}

# Download ETF data using yfinance
etf_data = yf.download(list(etf_portfolio.values()), start='1993-01-01', auto_adjust=True)
```

