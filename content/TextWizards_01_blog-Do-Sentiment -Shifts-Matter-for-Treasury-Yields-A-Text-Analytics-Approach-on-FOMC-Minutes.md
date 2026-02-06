---  
Title: Do Sentiment Shifts Matter for Treasury Yields? A Text Analytics Approach on FOMC Minutes (by Group "TextWizards")
Date: 2026-01-09 22:12  
Category: Reflective Report  
Tags: Group TextWizards
---
# Introduction
 Financial markets respond to monetary policy primarily through expectations about the future path of interest rates. In the sovereign debt market, this transmission mechanism is particularly direct: expectations of higher or more persistent policy rates drive bond prices down and yields up, while signals of easing support higher valuations. Consequently, the Federal Reserve’s policy stance, which is often classified along the spectrum of **hawkish and dovish**, serves as the primary anchor for Treasury market dynamics. To understand how these expectations are formed, it is important to consider the Federal Reserve’s communication process. Policy signals are released in multiple stages. On the meeting day, the policy statement communicates the interest rate decision and provides an initial signal to markets. This is followed by the Chair’s press conference, during which policymakers elaborate on their assessment of inflation, economic risks, and the future policy path. By the end of this stage, markets have already had considerable opportunity to update expectations. However, our study focuses on the critical third stage, which is the release of the **FOMC meeting minutes** approximately three weeks later. Unlike the policy statement or the press conference, the minutes do not introduce a new policy decision. Instead, they provide a detailed account of internal discussions, highlighting how different participants assess risks and trade-offs. We argue that **internal divergence** often serves as a precursor to future policy pivots, as it reveals the fragility of the prevailing consensus even when the headline policy stance remains unchanged. Besides, they may convey unexpected shifts in emphasis, such as heightened concern over inflation persistence or emerging downside risks to economic activity. These elements can prompt a reassessment of previously held expectations, even if the policy rate itself remains unchanged. The primary challenge, as we have identified in our research, is the qualitative nature of “Fedpeak.” The deliberate ambiguity and carefully balanced language used by policymakers make these nuances difficult to capture with traditional keyword-based tools. To overcome this, we utilize advanced Large Language Models (LLMs) to quantify the subtle narrative, transforming qualitative deliberations into a structured Divergence Score and Sentiment Delta.
 # Web Scraping

## FOMC Meeting Minutes Collection

The process starts by scraping the FOMC calendar page at `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`. A custom header is defined to avoid being identified as automated traffic. The following code demonstrates this process:

```python

#import packages

import pandas as pd

import numpy

import requests

import re

from bs4 import BeautifulSoup

import yfinance

# 1. Define constants

# FOMC Calendar URL

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

  

# Custom headers to avoid being blocked

HEADERS = {

'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',

'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',

'Accept-Language': 'en-US,en;q=0.5',

'Accept-Encoding': 'gzip, deflate',

'Connection': 'keep-alive',

}

```

After fetching the page using the `requests` package, BeautifulSoup parses the HTML to extract meeting minutes URLs. We need to identify links starting with `<a>` tags and containing 'fomcminutes' in `.htm` format. The links extracted from the page (`href`) are often relative paths (e.g., `/monetarypolicy/fomcminutes20230104.htm`). The code converts them to complete, working web addresses by prepending the base domain (`https://www.federalreserve.gov`). The following code demonstrates this process:

```python

# 2. Fetch and parse the webpage

response = requests.get(CALENDAR_URL, headers=HEADERS)

soup = BeautifulSoup(response.text, "html.parser")

links_data = []

# 3. Extract all relevant hyperlinks

for a in soup.find_all('a', href=True):

href = a['href']

# 4. Filter for FOMC minutes HTML files

if 'fomcminutes' in href and href.endswith('.htm'):

# 5. Extract meeting date from URL

date_match = re.search(r'(\d{8})', href)

if date_match:

date_str = date_match.group(1)

# 6. Construct complete URLs

BASE_URL = 'https://www.federalreserve.gov'

final_url = f"{BASE_URL}{href}"

# 7. Store meeting data

links_data.append({

"meeting_date": date_str,

"minutes_url": final_url

})

```
Our FOMC meeting minutes data collection covers the period from January 29, 2020 to December 10, 2025 (20200129 to 20251210), ensuring comprehensive coverage of recent monetary policy cycles. After deduplication and chronological sorting, the results are saved to `URL.csv`.

The following code demonstrates this process:

```python

# 8. Process and clean the data

df_links = (

pd.DataFrame(links_data)

.drop_duplicates(subset=['meeting_date']) # Remove duplicates

.sort_values("meeting_date", ascending=False) # Sort chronologically

)

df_links.to_csv('URL.csv')

```
## Financial Data from Yahoo Finance

We choose the 10-year US Treasury yield (`^TNX`) and US Dollar Index (`DX-Y.NYB`) as our financial data to capture market reactions following the release of meeting minutes. We utilize the `yfinance` library to efficiently download historical market data spanning from a specified start date (e.g., 2019-01-01) and extract the 'Close' prices for daily data of these two key tickers `^TNX` and `DX-Y.NYB`. The following code demonstrates this process:

```python

# 1. Define time range

start_date = "2019-01-01"

# 2. Download data (10-year Treasury yield + Dollar Index)

assets = yf.download(["^TNX", "DX-Y.NYB"], start=start_date)

# 3. Data structuring

# Extract only closing prices

data = assets['Close'].reset_index()

# 4. Rename columns

data.rename(columns={

"^TNX": "yield_10y",

"DX-Y.NYB": "usd_index",

"Date": "date"

}, inplace=True)

```

Then we calculate the daily change in the 10-year US Treasury yield, which quantifies market movements. The processed data is saved to `market_data_daily.csv`, ready for merging with textual analysis results. The following code demonstrates this process:

```python

# 5. Calculate daily yield changes (key: capturing market shocks)

data['yield_change'] = data['yield_10y'].diff()

  

# 6. Save data, prepare for merging with AI sentiment scores

data.to_csv("market_data_daily.csv", index=False)

print("Financial baseline data prepared! First 5 rows:")

print(data.head())

```

## Considerations for Future Work

The 10-year Treasury yield reflects broad macroeconomic forces including long-term inflation expectations, global risk sentiment, and broader central bank policies, which may obscure immediate reactions to FOMC meeting minutes details. Short-term yields are typically more sensitive to direct monetary policy expectations and could provide a clearer, more timely signal of the market's interpretation of FOMC communications. Therefore, based on our analysis results, we may switch to short-term yields for more precise policy impact measurement in future work.
# Text Extracing and Cleaning:

From the previous part, we have got two CSV files that contain:

- **the URL of each FOMC meeting minutes page**, saved in "URL.csv".

- **the daily historical market data during the time we focus**, saved in "market_data_daily.csv".

In this post, I document the second part of the project, which focuses on **extracting and cleaning textual data from Federal Open Market Committee (FOMC) meeting minutes**.

The goal of this part is simple but crucial —

to build a clean, structured text dataframe that can later be used efficiently for **NLP tasks such as sentiment analysis**.


## 1. Data Frame Setup:

We first load the data and convert the meeting date into a proper datetime format for later time-series analysis.

```python

df_url = pd.read_csv("URL.csv")

df_url["meeting_date"] = pd.to_datetime(

df_url["meeting_date"].astype(str),

format="%Y%m%d"

)

```

## 2. Extract the Main Content from Articles:

We design a function that:

1. Fetches the HTML page

2. Locates the main article content

3. Removes irrelevant elements (scripts, navigation bars, footers)

4. Converts the cleaned HTML into plain text

  

```python

def fetch_minutes_text(url):

r = requests.get(url, headers=HEADERS, timeout=30)

r.raise_for_status()

soup = BeautifulSoup(r.text, "lxml")

main = soup.find("div", id="article") or soup.body

for tag in main.find_all(

["script","style","nav","header","footer","aside"]

):

tag.decompose()

text = main.get_text("\n", strip=True)

text = re.sub(r"\n{3,}", "\n\n", text).strip()

return text

```

## 3. Processing and Store Text Data in different fields:

To process all meeting minutes URLs, I loop through the dataset and apply the text extraction function to each page.

And for each minutes, I store:

1. meeting date

2. source URL

3. cleaned text

4. word count (as a rough length indicator)

  

```python

rows = []
for _, r in df_url.iterrows():

try:

text = fetch_minutes_text(r["minutes_url"])

rows.append({

"meeting_date": r["meeting_date"],

"minutes_url": r["minutes_url"],

"text_clean": text,

"word_count": len(text.split())

})

time.sleep(0.2)

```

Also, to check the errors during the process, we use the followings to log the possible errors instead of dropping them directly.

```python

except Exception as e:

rows.append({

"meeting_date": r["meeting_date"],

"minutes_url": r["minutes_url"],

"text_clean": None,

"word_count": None,

"error": str(e)

})

```

## 4. Final Output

All results are merged into a DataFrame and exported as a UTF-8 encoded CSV file:


```python

df_minutes_text = pd.DataFrame(rows).sort_values("meeting_date")

df_minutes_text.to_csv(

"minutes_text.csv",

index=False,

encoding="utf-8-sig"

)

```
Now we can get **a "minutes_text.csv" file** that contains the cleaned text data for each minutes. And **it is organized as the order of "meeting_date, minutes_url, text_clean, word_count"**, which can be used for the next analysis.

  
  
Though the process works, we noticed **several things that can be improved**:

1. the output dataframe appears as **an extremely long and continous block of content**, where sections are difficult to distinguish and rapid navigation is challenging.

2. **FOMC minutes are inherently structured**, typically including presented members and the information of the scretary, which are useless for sentimental analysis.

Thus, we may need to upgrade our code to further cleaning the text data and screen out the unnecessary.
# Text sentiment analysis

After extracting the text from web links via BeautifulSoup, we perform sentiment analysis on the text. Long-text processing is a key strength of large language models (LLMs). However, since each meeting minutes document contains a large volume of text—ranging from 50,000 to 80,000 characters—the free quota provided by major AI platforms' APIs is generally less than 2,000,000 characters, which can be exhausted quickly. Additionally, API calls for services like ChatGPT come with relatively high costs. Eventually, we identified the **Spark Lite model API** of the Spark Large Model, which offers free and unlimited access, and we adopted it for text analysis.

 
The invocation method and text prompts for it are provided in the following code:

```python

import requests

import json

import re

# Please define SPARK_API_KEY and SPARK_URL before using

# SPARK_API_KEY = "Your Spark API Key"

# SPARK_URL = "Spark API Endpoint URL" 

def analyze_fomc_with_spark(text_content):

"""

Call Spark API to analyze Federal Reserve FOMC meeting minutes text

:param text_content: FOMC meeting minutes text content

:return: Analysis result dictionary (including hawkish, dovish, divergence, guidance)

"""

# Build precise prompt, force to return JSON format results

prompt = f"""

Please analyze the following Federal Reserve FOMC meeting minutes text and output the analysis results in standard JSON format in strict accordance with the following requirements:

1. hawkish: Hawkish intensity, value range 0-1, the higher the value, the stronger the hawkish tendency

2. dovish: Dovish intensity, value range 0-1, the higher the value, the stronger the dovish tendency

3. divergence: Committee divergence degree, value range 0-1, the higher the value, the greater the divergence among committee members

4. guidance: Next monetary policy action, must select only one from Hike (interest rate hike), Hold (maintain status quo), Cut (interest rate cut)

Notes:

- Must return standard JSON format, do not add any additional explanatory text

- The sum of hawkish and dovish values does not need to equal 1, but both must be within the range of 0-1

- guidance must be one of Hike, Hold, Cut, no other values are allowed

Text to be analyzed:

{text_content[:80000]} # Limit text length to avoid exceeding API token limits

"""

headers = {

'Authorization': SPARK_API_KEY,

'Content-Type': "application/json"

}

# Build request body (disable stream output for simplified processing)

body = {

"model": "4.0Ultra",

"user": "fomc_analyzer",

"messages": [{"role": "user", "content": prompt}],

"stream": False, # Disable stream output to get complete results directly

"temperature": 0.1 # Reduce randomness to ensure result stability

}

try:

# Send request

response = requests.post(

url=SPARK_URL,

json=body,

headers=headers,

timeout=60

)

response.raise_for_status()

# Parse response

result = response.json()

if "choices" in result and len(result["choices"]) > 0:

content = result["choices"][0]["message"]["content"]

# Extract JSON content (handle possible extra text)

json_match = re.search(r'\{[\s\S]*\}', content)

if json_match:

json_str = json_match.group()

analysis = json.loads(json_str)

# Validate and correct fields

validated_analysis = {

"hawkish": max(0.0, min(1.0, float(analysis.get("hawkish", 0.0)))),

"dovish": max(0.0, min(1.0, float(analysis.get("dovish", 0.0)))),

"divergence": max(0.0, min(1.0, float(analysis.get("divergence", 0.0)))),

"guidance": analysis.get("guidance", "Hold") if analysis.get("guidance") in ["Hike", "Hold", "Cut"] else "Hold"

}

return validated_analysis

# Default return value

return {"hawkish": 0.0, "dovish": 0.0, "divergence": 0.0, "guidance": "Hold"}

except json.JSONDecodeError as e:

print(f"JSON parsing failed: {e}")

return {"hawkish": 0.0, "dovish": 0.0, "divergence": 0.0, "guidance": "Hold"}

except Exception as e:

print(f"API call failed: {e}")

return {"hawkish": 0.0, "dovish": 0.0, "divergence": 0.0, "guidance": "Hold"}

```

The output results are as follows:

| date | hawkish | dovish | divergence | guidance |
|:--------|:---------|:--------|:------------|:---------|
| 20250129 | 0.3 | 0.2 | 0.1 | Hold |
| 20250319 | 0.3 | 0.4 | 0.2 | Hold |
| 20250507 | 0.4 | 0.3 | 0.2 | Hold |
| 20250618 | 0.3 | 0.4 | 0.2 | Hold |

Further construct **sentiment increment indicators** based on the above metrics, conduct **correlation analysis** with the volatility of bond yields, and explore whether the meeting minutes are consistent with market expectations. The specific construction of indicators and correlation analysis will be further completed in subsequent practice, and we will carry out more visualization work in the future.


