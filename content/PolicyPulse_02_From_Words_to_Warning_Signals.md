---
Title: PolicyPulse： From Words to Warning Signals
Date: 2026-01-21 
Category: Reflective Report
---

## Step 1 Data Collection

### 1.1. Introduction


*In this session, I will walk you through my journey of collecting central bank speech data from the Bank for International Settlements (BIS) website. This work was part of our PolicyPulse: From Words to Warning Signals, a project of financial text analytics and NLP. I will discuss the initial challenges, the pivots in approach, and the final solution that combined bulk downloads with an undocumented API.*

In our group project, we aimed to analyze the textual content of speeches delivered by central bank officials from around the world. The BIS website serves as a rich repository of such speeches, making it an ideal data source. My responsibility was to collect a comprehensive dataset spanning from 2008 to 2019, including metadata such as:

- Speech title
- Speaker
- Central bank (institution)
- Date
- Venue
- Full text
- PDF file size and page count
- Language
- Category tags

This blog documents the process, the hurdles, and the eventual solution that enabled us to gather this data efficiently.

---

### 1.2. Initial Approach: Direct Web Scraping

My first instinct was to scrape the BIS speeches index page:  `https://www.bis.org/cbspeeches/index.htm`

I noticed that the page allowed filtering by date, and the URL structure was predictable:
`https://www.bis.org/cbspeeches/index.htm?fromDate=01%2F01%2F2008&tillDate=01%2F02%2F2008&cbspeeches_page=2`

My plan was straightforward:

1. Dynamically generate URLs for each month and page within our date range.
2. Parse each page to extract links to individual speech pages.
3. Visit each speech page, scrape metadata, and download the PDF.

#### The Challenge: Dynamic Content Loading

However, when I used `requests` to fetch the HTML, I found that the speech links were missing from the source. After inspecting the page with browser developer tools, I realized that the table of speeches was loaded dynamically via JavaScript. This meant that static scraping with requests would not work.

At this point, I considered using **Selenium** to simulate a browser and interact with the page. While feasible, this approach would be slow and resource-intensive for a large date range (2008–2019), especially since each page displayed only 10 speeches.

Here’s a snippet from my notes at the time, outlining the initial plan:

1. Start from the filtered index page (with date range).

2. Dynamically crawl each page (using Selenium) to extract speech links.

3. For each speech, extract the unique code (e.g., "r251211d") from the link.

4. Construct an API URL: https://www.bis.org/api/documents/review/r251211d.json

5. Fetch metadata from the JSON API.

6. Download the PDF and extract text.

7. Combine all information into a structured format.


---

### 1.3. The Turning Point: Bulk Download and Hidden API

While exploring the BIS website further, I discovered a **bulk download page**:  `https://www.bis.org/cbspeeches/download.htm`

This page offered a CSV file (like `speeches_2008.csv`) containing a list of speeches with basic metadata:

- Text
- Title
- Speaker
- Date
- PDF URL

This was a game-changer. With permission from our professor, I downloaded this CSV, which covered all speeches from 2008 to 2019 in one go. However, the CSV was missing several important fields:

- Institution number and name
- Language
- PDF file size and page count
- Category (recurse_category)
- Source information

Fortunately, during my earlier exploration, I had noticed that each speech had an associated JSON file accessible via an **undocumented API**:
`https://www.bis.org/api/documents/review/{speech_code}.json`

Where `{speech_code}` is the unique identifier (e.g., `r251211d`). This API returned all the missing fields.

Thus, I devised a two-step strategy:

1. **Use the bulk CSV** to get the list of speeches and their basic metadata.
2. **Enrich each record** by calling the JSON API to fetch additional details.

---

### 1.4. Implementation: The Python Script

I wrote a Python script (`bis_processor.py`) to automate this process. Below, I’ll explain the key components.

#### 1.4.1 Setting Up the Environment

The script uses several libraries for HTTP requests, JSON handling, and progress tracking.

```python
import csv
import json
import requests
import sys
import os
from tqdm import tqdm
import argparse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
```

#### 1.4.2 Robust HTTP Requests
To handle network issues and avoid being blocked, I set up a session with retry logic and custom headers.

```python
# Define custom headers for HTTP requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json",
}

# Configure a session with retry logic
session = requests.Session()
retries = Retry(
    total=5,  
    backoff_factor=1,  
    status_forcelist=[500, 502, 503, 504],  # Retry on these HTTP status codes
)
session.mount("https://", HTTPAdapter(max_retries=retries))
```

#### 1.4.3 Extracting the Speech Code
From the PDF URL in the CSV, I extracted the unique speech code.

```python
def extract_unique_code(url):
    """Extract the unique code from the URL."""
    return url.split("/")[-1].split(".")[0]
```

#### 1.4.4 Fetching Document Details
This function calls the BIS API and parses the JSON response. It includes error handling and a fallback for a common URL pattern.
```python
def fetch_document_details(api_url):
    """Fetch the JSON data from the API and extract relevant details."""
    try:
        response = session.get(api_url, headers=HEADERS, timeout=10)  
        response.raise_for_status()
        data = response.json()
        authors = data.get("authors")
        author_ids = [author["id"] for author in authors] if authors else []
        return {
            "institutions": data.get("institutions"),
            "language": data.get("language"),
            "pdf_file_size": data.get("pdf_file_size"),
            "pdf_pages": data.get("pdf_pages"),
            "sources": data.get("sources"),
            "recurse_category": data.get("recurse_category"),
            "author_id": author_ids,
        }
    except requests.HTTPError as e:
        if response.status_code == 404:
            # Modify the URL to include an additional suffix and retry
            modified_api_url = api_url.replace(".json", ".a.json")
            try:
                response = session.get(modified_api_url, headers=HEADERS, timeout=10)
                response.raise_for_status()
                data = response.json()
                authors = data.get("authors")
                author_ids = [author["id"] for author in authors] if authors else []
                return {
                    "institutions": data.get("institutions"),
                    "language": data.get("language"),
                    "pdf_file_size": data.get("pdf_file_size"),
                    "pdf_pages": data.get("pdf_pages"),
                    "sources": data.get("sources"),
                    "recurse_category": data.get("recurse_category"),
                    "author_id": author_ids,
                }
            except requests.RequestException as retry_error:
                print(f"Error fetching data from {modified_api_url}: {retry_error}")
        else:
            print(f"Error fetching data from {api_url}: {e}")
    except requests.RequestException as e:
        print(f"Error fetching data from {api_url}: {e}")
    return None
```

#### 1.4.5 Mapping Institution Numbers to Names
The BIS API returns institution numbers, not names. I used a local `institutions.json` file, which was also from an undocumented API, to map these numbers to human-readable names.
```python
def load_institutions_map(file_path):
    """Load the institutions.json file and create a mapping of id to name."""
    with open(file_path, "r") as file:
        try:
            institutions_data = json.load(file)
            if not isinstance(institutions_data, dict):
                raise ValueError("Expected a dictionary in institutions.json")
            return {int(key): value["name"] for key, value in institutions_data.items() if "name" in value}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return {}
        except ValueError as e:
            print(f"Error with data structure in {file_path}: {e}")
            return {}

```

#### 1.4.6 Main Processing Loop
The script reads the CSV, iterates through each row, enriches the data via the API, and writes the output as JSON.

```python
def process_csv(csv_file_path, output_dir):
    """Process the CSV file and generate the final JSON data."""
    with open(csv_file_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    institutions_map = load_institutions_map(institutions_json_path)

    processed_data = []
    for row in tqdm(rows, desc="Processing rows"):
        # Remove empty keys from the row
        row = {key: value for key, value in row.items() if key.strip()}

        url = row.get("url")
        if not url:
            continue

        # Extract unique code and generate API link
        unique_code = extract_unique_code(url)
        api_url = f"https://www.bis.org/api/documents/review/{unique_code}.json"

        # Fetch document details from the API
        document_details = fetch_document_details(api_url)
        if document_details is None:
            continue

        institution_number = document_details["institutions"]

        # Handle cases where institution_number is a list
        if isinstance(institution_number, list):
            institution_names = [institutions_map.get(num, "Unknown") for num in institution_number]
            institution_name = ", ".join(institution_names)
        else:
            institution_name = institutions_map.get(institution_number, "Unknown")

        # Add new data to the row
        row["institution_number"] = institution_number
        row["institution_name"] = institution_name
        row["language"] = document_details["language"]
        row["pdf_file_size"] = document_details["pdf_file_size"]
        row["pdf_pages"] = document_details["pdf_pages"]
        row["sources"] = document_details["sources"]
        row["recurse_category"] = document_details["recurse_category"]
        row["author_id"] = document_details["author_id"]

        processed_data.append(row)

    # Generate output JSON file name based on input CSV file name
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_json_path = os.path.join(output_dir, f"{base_name}_processed.json")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the processed data as JSON
    with open(output_json_path, "w") as json_file:
        json.dump(processed_data, json_file, indent=4)

    print(f"Processed data saved to {output_json_path}")
```

#### 1.4.7 Running the Script
The script can be run from the command line:

```bash
python bis_processor.py speeches_2008.csv --output_dir Data_BIS/processed/
```


---

### 1.5. Reflections and Lessons Learned

#### 1.5.1 Challenges Overcome
- Dynamic Content: Initially, I underestimated the complexity of scraping a JavaScript-heavy site. The discovery of the bulk download page saved significant time and effort.

- Missing Data: The bulk CSV was incomplete, but the hidden API filled the gaps. This highlights the importance of exploring network requests when dealing with modern websites.

- Scalability: Downloading and processing thousands of speeches required robust error handling and retry logic. The requests.Session with retries proved invaluable.

- Data Consistency: Mapping institution numbers to names required a separate lookup file, which I obtained from the BIS website.

#### 1.5.2 Key Takeaways
- Always look for official data exports before writing a scraper. Many organizations provide bulk downloads.

- Inspect network traffic to uncover hidden APIs that can simplify data collection.

- Build resilience into your scripts with retries, timeouts, and informative logging.

- Use progress bars (like tqdm) for long-running tasks to maintain visibility.

---

### 1.6. Conclusion
This data collection exercise was a valuable lesson in adaptability. By pivoting from a direct scraping approach to a hybrid method (bulk download + API enrichment), I was able to gather a rich dataset efficiently. The final dataset includes over a decade of central bank speeches, ready for the next stages of text preprocessing and analysis.

The complete version of code, along with the rest of our project code, is available in our group’s GitHub repository.


## Step 2 Data preproperssing 

### 2.1 Introduction
The preprocessing phase transforms raw central bank speeches into clean, machine-readable text while calculating corresponding market reactions. This involves text cleaning, tokenization, financial data calculation, and merging to create structured datasets for machine learning.

### 2.2 Dependencies and Initialization
The code begins by importing necessary libraries and initializing NLTK resources for text processing.

```python
import json
import os
import re
import nltk
import pandas as pd
import yfinance as yf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta

# Step 0: Initialize NLTK & Core Functions
# Download NLTK dependencies (only if missing)
def download_nltk_corpora():
    """Download required NLTK corpora (punkt_tab, stopwords) if not installed"""
    required_corpora = ['punkt_tab', 'stopwords']
    for corpus in required_corpora:
        try:
            nltk.data.find(f'corpora/{corpus}')
        except LookupError:
            nltk.download(corpus)

download_nltk_corpora()
```

### 2.3 Text Cleaning Functions

#### 2.3.1 Stopwords Configuration
The get_stopwords function combines NLTK's default English stopwords with financial domain-specific terms to filter low-information content.

```python
# Step 0.5: Define stopwords for text cleaning
def get_stopwords():
    """Combine NLTK default stopwords + financial speech-specific stopwords"""
    nltk_stopwords = set(stopwords.words('english'))
    financial_stopwords = {'speech', 'remarks', 'statement', 'thank', 'ladies', 'gentlemen'}
    return nltk_stopwords.union(financial_stopwords)
```

#### 2.3.2 Text Cleaning Function
The clean_text function removes structural noise, normalizes text, and filters stopwords to extract meaningful policy language from messy transcripts.

```python
# Step 1 : Text cleaning function (only return cleaned text, no original text)
def clean_text(text):
    """
    Clean BIS speech text: remove noise, stopwords, and non-meaningful terms
    Returns only cleaned text (no original text retained)
    """
    if not text or pd.isna(text):
        return ""
    
    # 1.1.  Remove table/chart markers, page numbers, URLs/emails
    text = re.sub(r'\|.*?\|+.*?(?:\n|$)', ' ', text)  # Remove table structures
    text = re.sub(r'(chart|table)\s+\d+(?:\.\d+)?(?::)?.*?(?:\n|$)', ' ', text, flags=re.IGNORECASE)  # Remove chart/table labels
    text = re.sub(r'(page|p)\.?\s+\d+(?:-\d+)?', ' ', text, flags=re.IGNORECASE)  # Remove page numbers
    text = re.sub(r'https?://\S+|www\.\S+|mailto:\S+|\S+@\S+', ' ', text)  # Remove URLs/emails
    text = re.sub(r'\[.*?\]', ' ', text)  # Remove [Applause], [Inaudible], etc.
    
    # 1.2.  Normalize symbols and remove numbers
    text = re.sub(r'[-—–]+', ' ', text)  # Normalize dashes/hyphens
    text = re.sub(r'\d+(?:,\d{3})*\.?\d*%?|%?\d+\.?\d*|[\d.eE+-]+', ' ', text)  # Remove all numbers/percentages
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Retain only letters and spaces
    
    # 1.3.  Filter stopwords and short terms
    text = re.sub(r'\s+', ' ', text).lower().strip()
    stop_words = get_stopwords()
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words and len(word) > 2]
    cleaned_text = re.sub(r'\s+', ' ', ' '.join(filtered_words)).strip()
    
    return cleaned_text
```

### 2.4 Financial Data Calculation
The get_financial_future_3m_change function fetches VIX and S&P 500 data and calculates 90-day future percentage changes for market reaction labeling.

```python
#  Step 2: Calculate 3-Month Future Change for VIX/SP500
def get_financial_future_3m_change(year):
    """
    Fetch daily VIX (^VIX) and SP500 (^GSPC) data for a given year, calculate 3-month (90-day) future percentage change
    Formula: (Price at t+90 days / Price at t - 1) * 100 (percentage change)
    Args:
        year (int): Target year (2008-2019)
    Returns:
        pd.DataFrame: Daily data with columns: Date, vix_future_3m_change, sp500_future_3m_change
    """
    # Define date range: extend to +90 days beyond the year to calculate future changes for Dec dates
    start_date = f"{year}-01-01"
    end_date = f"{year+1}-03-31"  # Ensure enough data for Dec's 3-month change
    
    # Fetch daily data for VIX and SP500
    tickers = ["^VIX", "^GSPC"]
    try:
        # Download VIX Data
        df_vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d")
        df_vix = df_vix[["Close"]].rename(columns={"Close": "vix"})
        
        # Download SP500 Data
        df_sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval="1d")
        sp_col = "Adj Close" if "Adj Close" in df_sp500.columns else "Close"
        df_sp500 = df_sp500[[sp_col]].rename(columns={sp_col: "sp500"})
        
        # Merging
        df_fin = pd.merge(df_vix, df_sp500, left_index=True, right_index=True, how="outer")
        df_fin = df_fin.reset_index()  # Convert index to Date column
        df_fin.columns = ["Date", "vix", "sp500"]  # Rename columns for clarity
    except Exception as e:
        print(f"❌ fail to download data of（{year}）: {str(e)}")
        return pd.DataFrame()  
    
    # Convert Date to datetime and set as index (for shift calculation)
    df_fin["Date"] = pd.to_datetime(df_fin["Date"])
    df_fin = df_fin.set_index("Date")
    
    # Calculate 3-month (90-day) future change (percentage)
    # shift(-90) = get price 90 days in the future; fillna with NaN for missing future data
    df_fin = df_fin.fillna(method="ffill")  
    df_fin["vix_future_3m_change"] = (df_fin["vix"].shift(-90) / df_fin["vix"] - 1) * 100
    df_fin["sp500_future_3m_change"] = (df_fin["sp500"].shift(-90) / df_fin["sp500"] - 1) * 100
    
    # Reset index and filter back to the target year (remove extended dates)
    df_fin = df_fin.reset_index()
    df_fin["Year"] = df_fin["Date"].dt.year
    df_fin = df_fin[df_fin["Year"] == year].drop(columns=["Year", "vix", "sp500"])
    
    # Round to 2 decimal places for readability
    df_fin[["vix_future_3m_change", "sp500_future_3m_change"]] = df_fin[["vix_future_3m_change", "sp500_future_3m_change"]].round(2)
    
    return df_fin
```

### 2.5 Data Merging Pipeline
The process_yearly_data function integrates cleaned speech text with financial market data, aligning each speech with corresponding market outcomes.

```python
#  Step 3: Process Yearly Speech + Financial Data 
def process_yearly_data(year, speech_data_dir, output_dir):
    """
    Process a single year's speech data + financial data:
    1. Load yearly speech file (e.g., speeches_2008_processed.json)
    2. Clean text (only retain cleaned text)
    3. Match with VIX/SP500 3-month future change by date
    4. Save as merged_data_YYYY.json
    Args:
        year (int): Target year (2008-2019)
        speech_data_dir (str): Directory containing yearly speech JSON files
        output_dir (str): Directory to save merged JSON files
    """
    # 3.1. Load yearly speech file
    speech_file = os.path.join(speech_data_dir, f"speeches_{year}_processed.json")
    if not os.path.exists(speech_file):
        print(f"⚠️ Skipping {year}: Speech file not found ({speech_file})")
        return
    
    with open(speech_file, "r", encoding="utf-8") as f:
        speech_data = json.load(f)
    
    # Convert to DataFrame and clean
    df_speeches = pd.json_normalize(speech_data)
    
    # 3.2. Process speech data (date + clean text)
    df_speeches["date"] = pd.to_datetime(df_speeches["date"], errors="coerce")
    # Keep only title/date/text (filter missing values)
    df_speeches = df_speeches[["title", "date", "text"]].dropna(subset=["title", "date", "text"])
    # Clean text (replace original text with cleaned text)
    df_speeches["text"] = df_speeches["text"].apply(clean_text)
    # Filter empty cleaned text
    df_speeches = df_speeches[df_speeches["text"] != ""].reset_index(drop=True)
    
    if len(df_speeches) == 0:
        print(f"⚠️ Skipping {year}: No valid cleaned speech data")
        return
    
    # 3.3. Fetch financial data (VIX/SP500 3-month future change)
    df_fin = get_financial_future_3m_change(year)
    
    if df_fin.empty:
        print(f"⚠️ Skipping {year}: No valid financial data")
        return
    
    # 3.4. Match speech data with financial data by date (handle non-trading days)
    # Convert speech dates to date-only (remove time)
    df_speeches["date"] = df_speeches["date"].dt.date
    df_fin["Date"] = df_fin["Date"].dt.date
    
    # Merge (left join to retain all speeches; fill NaN for missing financial data)
    df_merged = pd.merge(
        df_speeches,
        df_fin,
        left_on="date",
        right_on="Date",
        how="left"
    ).drop(columns=["Date"])
    
    # Rename columns to match your requirement
    df_merged = df_merged.rename(columns={
        "title": "title",
        "date": "date",
        "text": "text",
        "vix_future_3m_change": "vix_future_3m_change",
        "sp500_future_3m_change": "sp500_future_3m_change"
    })
    
    # Convert date back to string (JSON-compatible)
    df_merged["date"] = df_merged["date"].astype(str)
    # Replace NaN with "N/A" for missing financial data (more readable in JSON)
    df_merged = df_merged.fillna("N/A")
    
    # 3.5. Save merged data as JSON
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"merged_data_{year}.json")
    df_merged.to_json(
        output_file,
        orient="records",
        force_ascii=False,
        indent=2
    )
    
    print(f"✅ Processed {year}: {len(df_merged)} valid speeches saved to {output_file}")
```

### 2.6 Main Execution Loop
The main execution script processes all years from 2008 to 2019 sequentially, creating annual merged datasets.

```python
# Step 4: Main Execution (Traverse 2008-2019) 
if __name__ == "__main__":
    # Configuration (ADJUST THESE PATHS TO YOUR ENVIRONMENT!)
    SPEECH_DATA_DIR = r"BIS"  # Directory containing speeches_2008_processed.json, etc.
    OUTPUT_DIR = r"BIS_Merged_Data"  # Output directory for merged_data_YYYY.json
    
    # Traverse 2008 to 2019
    for year in range(2008, 2020):  # 2020 is exclusive (stops at 2019)
        process_yearly_data(year, SPEECH_DATA_DIR, OUTPUT_DIR)
    
    print("\n All years processed! Check output directory: ", OUTPUT_DIR)
```

### 2.7 Output Structure
The preprocessing pipeline generates structured JSON data with five core fields per document, providing clean text paired with market indicators.

{
  "title": "Speech Title",
  "date": "2019-06-15",
  "text": "cleaned text content here",
  "vix_future_3m_change": 12.34,
  "sp500_future_3m_change": -5.67
}


## Step 3  Dictionary

### 3.1 Introduction 

In dictionary construction, in addition to using the AFINN predefined sentiment dictionary as an external reference, we further build a custom dictionary entirely derived from the sample data itself.

The motivation for this approach is that general-purpose sentiment dictionaries are not designed specifically for financial contexts or for particular text sources. As a result, their sentiment labels may not accurately reflect the market-relevant meaning of words in our setting.

Therefore, we adopt a data-driven approach, allowing a word’s “sentiment” or market meaning to be determined directly by the historical market outcomes associated with its occurrences, rather than by ex ante human annotation.

The central idea of the custom dictionary can be summarized as follows:

* If a word tends to be followed by relatively consistent market reactions after it appears in texts, then that word can be regarded as carrying a specific market meaning.

* Unlike traditional sentiment dictionaries, the “meaning” here does not refer simply to positive or negative emotion. Instead, it refers to statistical characteristics related to future market behavior, such as:

* Whether the word is more frequently associated with future market increases or declines

* Whether it is related to rising or falling volatility

* Whether it tends to appear only during extreme market conditions

### 3.2 Building Dictionary

#### Step 1: Constructing Future-Looking Market Labels

To ensure that dictionary construction does not confound the causal ordering, we first construct future-looking market change indicators for each date, measuring market performance after the text appears—specifically over the next three months (approximately 63 trading days).

Concretely, we compute:

* The three-month forward return of the S&P 500

* The three-month forward change of the VIX

This design ensures that all statistical properties assigned to each word are based on post-occurrence market outcomes, rather than contemporaneous or backward-looking information. These labels are constructed using the dataset established earlier.

#### Step 2: Building the “Word–Date–Market Outcome” Mapping

In text processing, we adopt unigrams (individual words) as the most basic and robust analytical unit.
Each document is first tokenized into a list of unigrams and then converted into a set of unique words, preventing multiple counts of the same word within a single document.

As we iterate through the documents, we record three core pieces of information for each word:

The set of distinct dates on which the word appears

The corresponding future S&P 500 change for each date

The corresponding future VIX change for each date

Importantly, we use “word × date” as the minimal counting unit, rather than raw word frequency. The implicit assumption is that:

What matters is whether a word appears on a given day, not how many times it appears.

This treatment effectively mitigates distortions caused by differences in document length or excessive repetition of high-frequency words, leading to a more stable and interpretable dictionary construction.

```python
import pandas as pd
from collections import defaultdict
import numpy as np


def calculate_future_changes(series, days=63):
    future_values = series.shift(-days)
    return (future_values - series) / series


df_market['sp500_change_3m'] = calculate_future_changes(df_market['sp500_close'], 63)
df_market['vix_change_3m'] = calculate_future_changes(df_market['vix_close'], 63)


word_market_changes = defaultdict(lambda: {
    'dates': set(),  
    'sp500_changes': {},
    'vix_changes': {}
})


for idx, row in df_clean.iterrows():
    date = idx  
    
    unigrams = row['unigrams']
    
    
    if hasattr(date, 'date'):
        date_key = date.date()  
    else:
        date_key = date
    
    if date_key in df_market.index:
        sp500_change = df_market.loc[date_key, 'sp500_change_3m']
        vix_change = df_market.loc[date_key, 'vix_change_3m']
        
        
        unique_words = set(unigrams)
        
        
        for word in unique_words:
            word_market_changes[word]['dates'].add(date_key)
            word_market_changes[word]['sp500_changes'][date_key] = sp500_change
            word_market_changes[word]['vix_changes'][date_key] = vix_change

word_data = []
for word, changes in word_market_changes.items():
    if changes['dates']:  
        dates_list = list(changes['dates'])
        sp500_values = [changes['sp500_changes'][d] for d in dates_list if pd.notna(changes['sp500_changes'][d])]
        vix_values = [changes['vix_changes'][d] for d in dates_list if pd.notna(changes['vix_changes'][d])]
        
        word_data.append({
            'word': word,
            'dates': dates_list,
            'sp500_changes': sp500_values,
            'vix_changes': vix_values,
            'unique_dates_count': len(dates_list), 
            'avg_sp500_change': np.mean(sp500_values) if sp500_values else None,
            'avg_vix_change': np.mean(vix_values) if vix_values else None,
        })

df_word_market = pd.DataFrame(word_data)

df_word_market = df_word_market.sort_values('unique_dates_count', ascending=False)
```

#### Step 3: Constructing an “Extended Dictionary” with Robustification

In further analysis, we recognize that simple averages are highly sensitive to noise and extreme values. Therefore, in the extended version, we introduce multiple robust statistical dimensions into the custom dictionary, transforming it from a single-number representation into a multi-metric structure.

Specifically, the extensions include:

(1) Directional information (binarization)
Future market changes are converted into directional indicators (up / down), emphasizing the consistency between a word’s occurrence and subsequent market direction.

(2) Threshold filtering
Market movements are included in the statistics only when they exceed predefined thresholds, allowing the analysis to focus on economically meaningful fluctuations.

(3) Joint indicators (S&P 500 + VIX)
By jointly incorporating equity index and volatility information, we capture a word’s association with the overall market state, rather than with a single market dimension.

(4) Frequency thresholds
Statistical results are retained only for words that appear on a sufficient number of distinct dates, reducing the risk of spurious correlations driven by low-frequency words.

Through these procedures, we ultimately construct a hierarchical and adaptive dictionary system, in which each word is characterized not only by whether it is relevant, but also by under what conditions it is relevant.

```python

import pandas as pd
from collections import defaultdict
import numpy as np

# Compute future market changes over a specified horizon (default: 3 months ≈ 63 trading days)
def calculate_future_changes(series, days=63):
    """Compute percentage change over a future window"""
    future_values = series.shift(-days)
    return (future_values - series) / series

# Compute 3-month forward changes for S&P 500 and VIX
df_market['sp500_change_3m'] = calculate_future_changes(df_market['sp500_close'], 63)
df_market['vix_change_3m'] = calculate_future_changes(df_market['vix_close'], 63)

# Create a mapping from words to future market changes
word_market_changes = defaultdict(lambda: {
    'dates': set(),              # Use a set instead of a list to automatically remove duplicates
    'sp500_changes': {},
    'vix_changes': {},
    'sp500_binary': {},          # New: binarized S&P 500 changes
    'vix_binary': {},            # New: binarized VIX changes
})

# Threshold parameters
sp500_threshold = 0.03           # 3% threshold (adjustable)
vix_threshold = 0.10             # 10% threshold (adjustable)
frequency_threshold = 10         # Minimum word frequency threshold (adjustable)

# Iterate through each document — date is the index
for idx, row in df_clean.iterrows():
    date = idx                   # Date is stored in the index, not a column
    unigrams = row['unigrams']

    # Ensure date format matches df_market index
    if hasattr(date, 'date'):
        date_key = date.date()   # Convert to date object
    else:
        date_key = date

    # Check whether market data exists for this date
    if date_key in df_market.index:
        sp500_change = df_market.loc[date_key, 'sp500_change_3m']
        vix_change = df_market.loc[date_key, 'vix_change_3m']

        # Extract unique words from the document (deduplicated)
        unique_words = set(unigrams)

        # Record market changes for each unique word in the document
        for word in unique_words:
            word_market_changes[word]['dates'].add(date_key)
            word_market_changes[word]['sp500_changes'][date_key] = sp500_change
            word_market_changes[word]['vix_changes'][date_key] = vix_change

            # 1. Binarize changes (-1, 0, +1)
            if pd.notna(sp500_change):
                if sp500_change > 0:
                    word_market_changes[word]['sp500_binary'][date_key] = 1
                elif sp500_change < 0:
                    word_market_changes[word]['sp500_binary'][date_key] = -1
                else:
                    word_market_changes[word]['sp500_binary'][date_key] = 0

            if pd.notna(vix_change):
                # VIX is typically inversely related to equity markets
                if vix_change > 0:
                    word_market_changes[word]['vix_binary'][date_key] = 1   # Rising VIX usually implies market stress
                elif vix_change < 0:
                    word_market_changes[word]['vix_binary'][date_key] = -1  # Falling VIX usually implies market recovery
                else:
                    word_market_changes[word]['vix_binary'][date_key] = 0

# Convert aggregated results to DataFrame
word_data = []
for word, changes in word_market_changes.items():
    if changes['dates']:  # Keep only words with valid observations
        # Convert date set to list for computation
        dates_list = list(changes['dates'])

        # Base continuous values
        sp500_values = [
            changes['sp500_changes'][d]
            for d in dates_list
            if pd.notna(changes['sp500_changes'][d])
        ]
        vix_values = [
            changes['vix_changes'][d]
            for d in dates_list
            if pd.notna(changes['vix_changes'][d])
        ]

        # 1. Binarized averages
        sp500_binary_vals = [
            changes['sp500_binary'][d]
            for d in dates_list
            if d in changes['sp500_binary']
        ]
        vix_binary_vals = [
            changes['vix_binary'][d]
            for d in dates_list
            if d in changes['vix_binary']
        ]

        # 2. Threshold-filtered averages
        sp500_values_thresh = [x for x in sp500_values if abs(x) > sp500_threshold]
        vix_values_thresh = [x for x in vix_values if abs(x) > vix_threshold]

        # 3. Joint scores (when both S&P 500 and VIX exhibit abnormal behavior)
        joint_scores = []
        joint_division_scores = []
        for d in dates_list:
            if d in changes['sp500_changes'] and d in changes['vix_changes']:
                sp_val = changes['sp500_changes'][d]
                vix_val = changes['vix_changes'][d]
                if pd.notna(sp_val) and pd.notna(vix_val):
                    # Sum-based joint score (VIX sign reversed)
                    joint_sum = sp_val + (-vix_val)
                    joint_scores.append(joint_sum)

                    # Ratio-based joint score (avoid division by zero)
                    if vix_val != 0:
                        joint_div = sp_val / abs(vix_val)
                        joint_division_scores.append(joint_div)

        # 4. Frequency-threshold-adjusted averages
        # Compute word frequency (number of unique dates)
        word_frequency = len(dates_list)

        # Base averages
        avg_sp500 = np.mean(sp500_values) if sp500_values else None
        avg_vix = np.mean(vix_values) if vix_values else None

        # Use averages only if frequency exceeds the threshold
        avg_sp500_freq_thresh = avg_sp500 if word_frequency >= frequency_threshold else None
        avg_vix_freq_thresh = avg_vix if word_frequency >= frequency_threshold else None

        word_data.append({
            'word': word,
            'dates': dates_list,
            'sp500_changes': sp500_values,
            'vix_changes': vix_values,
            'unique_dates_count': len(dates_list),  # Number of distinct dates the word appears

            # Base averages
            'avg_sp500_change': avg_sp500,
            'avg_vix_change': avg_vix,

            # 1. Binarized variables
            'avg_sp500_binary': np.mean(sp500_binary_vals) if sp500_binary_vals else None,
            'avg_vix_binary': np.mean(vix_binary_vals) if vix_binary_vals else None,

            # 2. Threshold-filtered averages
            'avg_sp500_threshold': np.mean(sp500_values_thresh) if sp500_values_thresh else None,
            'avg_vix_threshold': np.mean(vix_values_thresh) if vix_values_thresh else None,
            'count_sp500_threshold': len(sp500_values_thresh),  # Observations exceeding threshold
            'count_vix_threshold': len(vix_values_thresh),

            # 3. Joint scores
            'avg_joint_sum': np.mean(joint_scores) if joint_scores else None,
            'avg_joint_division': np.mean(joint_division_scores) if joint_division_scores else None,
            'joint_observations': len(joint_scores),  # Number of joint observations

            # 4. Frequency-threshold-adjusted averages
            'avg_sp500_freq_thresh': avg_sp500_freq_thresh,
            'avg_vix_freq_thresh': avg_vix_freq_thresh,
            'meets_frequency_threshold': word_frequency >= frequency_threshold,
        })

df_word_market = pd.DataFrame(word_data)

# Sort by the number of unique dates (descending)
df_word_market = df_word_market.sort_values('unique_dates_count', ascending=False)
```

#### Step 4: Mapping the AFINN Dictionary to the Sample Vocabulary Space

In implementation, we first map the AFINN sentiment scores onto the set of words that actually appear in our sample.
For each word, if it exists in the AFINN dictionary, it is assigned the corresponding sentiment score; otherwise, its score is set to zero.

 ```python
 df_word_market = df_word_market.assign(
    afinn_score=df_word_market['word'].map(
        lambda x: afinn.loc[afinn['word'] == x, 'score'].values[0] 
        if x in afinn['word'].values else 0
    )
)
 ```
 
#### Step 5 From Word-Level Dictionaries to Document-Level Scores
 
 After constructing the dictionaries, we face a practical question:
how should word-level information be aggregated to the document level, so that it can be aligned with market variables or other units of analysis?

To address this, we adopt a simple and interpretable aggregation scheme.
For each document, we compute the average score of all words appearing in the text.

Specifically, for each dictionary dimension (including AFINN scores, custom market-based scores, and others), we calculate:

* The mean score of all words appearing in the document

* Words that do not exist in a given dictionary are assigned a score of zero
 
```python
import pandas as pd
import numpy as np

# 1. Ensure df_clean has the correct columns
print("Columns in df_clean:", df_clean.columns.tolist())
print("Index name of df_clean:", df_clean.index.name)

# 2. If the date is stored in the index, convert it to a column
if df_clean.index.name == 'date' or isinstance(df_clean.index, pd.DatetimeIndex):
    df_clean = df_clean.reset_index()
    print("Date index has been converted to a column")

# 3. Check whether a 'date' column exists
if 'date' not in df_clean.columns:
    # Try to find alternative column names that may represent dates
    date_cols = [col for col in df_clean.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        print(f"Potential date column(s) found: {date_cols}")
        # Rename the first candidate to 'date'
        df_clean = df_clean.rename(columns={date_cols[0]: 'date'})
    else:
        # If no date column is found, create one using the index
        print("No date column found, using index as date")
        df_clean['date'] = df_clean.index

# 4. Select numeric score columns only
numeric_score_columns = []
for col in df_word_market.columns:
    if col in ['word', 'dates', 'unique_dates_count']:
        continue

    # Check data type
    if df_word_market[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
        numeric_score_columns.append(col)
    else:
        # Attempt type conversion
        try:
            df_word_market[col] = pd.to_numeric(df_word_market[col], errors='coerce')
            numeric_score_columns.append(col)
        except:
            print(f"Skipping column {col}: cannot be converted to numeric")

print(f"Using {len(numeric_score_columns)} numeric score columns")
print("Score columns:", numeric_score_columns)

# 5. Create dictionaries using numeric score columns only
word_score_dicts = {}
for col in numeric_score_columns:
    word_score_dicts[col] = dict(
        zip(
            df_word_market['word'],
            df_word_market[col].fillna(0)
        )
    )

# 6. Simplified document-level score calculation function
def calculate_document_scores_simple(text, word_score_dicts):
    """Compute mean scores for a single document"""
    if not isinstance(text, str) or not text.strip():
        return {f"{col}_mean": 0.0 for col in word_score_dicts.keys()}

    words = text.lower().split()
    if len(words) == 0:
        return {f"{col}_mean": 0.0 for col in word_score_dicts.keys()}

    result = {}

    for col, score_dict in word_score_dicts.items():
        scores = []
        for word in words:
            if word in score_dict:
                scores.append(score_dict[word])

        if scores:
            result[f"{col}_mean"] = float(np.mean(scores))
        else:
            result[f"{col}_mean"] = 0.0

    return result

# 7. Compute document scores
print("Starting document score computation...")
results = []

for idx, row in df_clean.iterrows():
    # Safely extract fields
    date = row.get('date', pd.NaT)
    title = row.get('title', '')
    text = row.get('text', '')

    # Calculate document scores
    doc_scores = calculate_document_scores_simple(text, word_score_dicts)

    # Construct result row
    result_row = {
        'date': date,
        'title': title,
        'text': str(text)[:300] + "..." if len(str(text)) > 300 else str(text),
    }

    # Append computed scores
    result_row.update(doc_scores)
    results.append(result_row)

    # Progress update
    if idx > 0 and idx % 1000 == 0:
        print(f"Processed {idx} documents")

# 8. Convert results to DataFrame
df_document_scores = pd.DataFrame(results)

# 9. Reorder columns
base_columns = ['date', 'title', 'text']
other_columns = [col for col in df_document_scores.columns if col not in base_columns]

# Sort by importance
if 'avg_sp500_change_mean' in other_columns:
    other_columns.remove('avg_sp500_change_mean')
    other_columns.insert(0, 'avg_sp500_change_mean')

# Move other important columns to the front
important_cols = ['avg_vix_change_mean', 'avg_sp500_binary_mean', 'afinn_score_mean']
for col in important_cols:
    if col in other_columns:
        other_columns.remove(col)
        other_columns.insert(0, col)

final_columns = base_columns + other_columns
df_document_scores = df_document_scores[final_columns]

df_document_scores.head(3)
```
 
## 4. Modeling: XGBOOST

### 4.1. Motivation and Problem Setup

After transforming our speeches into document-level scores (Step 3), the next question was straightforward but important:

Do these text-derived signals contain predictive information about future market stress?

To make the evaluation concrete, we framed this as a binary classification task. Our target variable is:

sp500_drop_8pct_next_3m: whether the S&P 500 experiences a drop of at least 8% within the next three months.

This definition aligns with our project goal of moving “from words to warning signals”. Instead of predicting small fluctuations, we focus on tail-risk style drawdowns that are economically meaningful.


### 4.2. Time-Based Train–Test Split (Avoiding Look-Ahead Bias)

A key modeling decision was to split the dataset by time, rather than randomly. In finance, random splitting can easily introduce look-ahead bias and inflate performance because the model may implicitly learn future patterns.

We therefore use:

* Training set: 2008–2014

* Test set: 2015–2019

This mimics a realistic deployment setting: the model learns from past speeches and is evaluated on a later period it has never seen.

To ensure the split is valid, we first check:

* the date range

* the number of speeches per year

* the positive class proportion per year (crucial for crash prediction tasks)

This step also helped us detect potential issues early (e.g., missing years or extremely imbalanced periods).

### 4.3. Feature Construction: Combining Dictionary Scores and TF–IDF

Our features come from two sources:

(1) Numeric features: dictionary-based document scores

These are the outputs from Step 3, including:

* custom market-based scores (e.g., avg_sp500_change_mean, avg_vix_change_mean)

* robustified variants (binary / threshold / joint scores, if included)

* AFINN-based sentiment signal (afinn_score_mean)

This set preserves interpretability: each feature corresponds to a clear “text → market” mapping mechanism.

(2) Text features: TF–IDF (lightweight baseline)

In addition, we add a small TF–IDF representation of (text + title):

max_features = 50 (to control dimensionality)

min_df = 2 (to reduce extreme sparsity)

stop_words = 'english'

We treat TF–IDF as a sanity check: if a simple bag-of-words representation adds incremental predictive power beyond our dictionary signals, that suggests our dictionary is capturing only part of the information.

Importantly, the TF–IDF vectorizer is fit on the training set only, then applied to the test set, to avoid leakage.

```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
import joblib

# Assume df_document_scores is your existing DataFrame
# Check basic data information
print("Basic Data Information:")
print(f"Dataset shape: {df_document_scores.shape}")
print(f"Column names: {df_document_scores.columns.tolist()}")

# 1. Data Preprocessing
# Ensure the date column is in datetime format
df_document_scores['date'] = pd.to_datetime(df_document_scores['date'], errors='coerce')

# Check date range
print(f"\nData Date Range: {df_document_scores['date'].min()} to {df_document_scores['date'].max()}")

# Extract year from date
df_document_scores['year'] = df_document_scores['date'].dt.year

# Inspect data distribution by year
print("\nData Distribution by Year:")
year_counts = df_document_scores['year'].value_counts().sort_index()
print(year_counts)

# Check whether data exists for 2008–2014 (training) and 2015–2019 (testing)
required_train_years = list(range(2008, 2015))
required_test_years = list(range(2015, 2020))

train_years_present = [year for year in required_train_years if year in df_document_scores['year'].values]
test_years_present = [year for year in required_test_years if year in df_document_scores['year'].values]

print(f"\nTrain years present (2008-2014): {len(train_years_present)}/{len(required_train_years)}")
print(f"Test years present (2015-2019): {len(test_years_present)}/{len(required_test_years)}")

# Handle missing values
df_cleaned = df_document_scores.copy()

# For numeric features, fill missing values with the median (robust for financial data)
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'year':  # Exclude year column
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# For text columns, fill missing values with empty strings
text_cols = ['title', 'text']
for col in text_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].fillna('')

# 2. Time-based Train-Test Split
# Training set: 2008–2014
# Test set: 2015–2019
train_data = df_cleaned[df_cleaned['year'].between(2008, 2014)].copy()
test_data = df_cleaned[df_cleaned['year'].between(2015, 2019)].copy()

print(f"\nTraining set (2008-2014) size: {train_data.shape}")
print(f"Test set (2015-2019) size: {test_data.shape}")

if len(train_data) == 0:
    raise ValueError("Training set (2008-2014) is empty, cannot train model!")
if len(test_data) == 0:
    raise ValueError("Test set (2015-2019) is empty, cannot evaluate model!")

# Check year distribution in training and test sets
print("\nTraining set year distribution:")
print(train_data['year'].value_counts().sort_index())

print("\nTest set year distribution:")
print(test_data['year'].value_counts().sort_index())

# 3. Feature Engineering
# Select numeric features (excluding target, date, and year)
exclude_cols = ['date', 'year', 'sp500_drop_8pct_next_3m', 'title', 'text']
numeric_features = [
    col for col in df_cleaned.columns
    if col not in exclude_cols and df_cleaned[col].dtype in [np.int64, np.float64]
]

print(f"\nNumber of numeric features used: {len(numeric_features)}")
print("Numeric features:", numeric_features)

# Create text-based features if text data is available
if 'text' in df_cleaned.columns and 'title' in df_cleaned.columns:
    print("\nCreating text features...")
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english', min_df=2)

    # Training text
    train_text = train_data['text'].astype(str) + ' ' + train_data['title'].astype(str)
    train_text_features = vectorizer.fit_transform(train_text)
    train_text_features_df = pd.DataFrame(
        train_text_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(train_text_features.shape[1])]
    )

    # Test text
    test_text = test_data['text'].astype(str) + ' ' + test_data['title'].astype(str)
    test_text_features = vectorizer.transform(test_text)
    test_text_features_df = pd.DataFrame(
        test_text_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(test_text_features.shape[1])]
    )

    # Combine numeric and text features
    X_train_numeric = train_data[numeric_features].reset_index(drop=True)
    X_train = pd.concat([X_train_numeric, train_text_features_df], axis=1)

    X_test_numeric = test_data[numeric_features].reset_index(drop=True)
    X_test = pd.concat([X_test_numeric, test_text_features_df], axis=1)
else:
    # Use numeric features only
    X_train = train_data[numeric_features].reset_index(drop=True)
    X_test = test_data[numeric_features].reset_index(drop=True)

# Target variable
y_train = train_data['sp500_drop_8pct_next_3m'].reset_index(drop=True)
y_test = test_data['sp500_drop_8pct_next_3m'].reset_index(drop=True)

print(f"\nTraining features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}")

# Check class distribution
print(f"\nTraining positive class proportion: {y_train.mean():.2%} ({y_train.sum()}/{len(y_train)})")
print(f"Test positive class proportion: {y_test.mean():.2%} ({y_test.sum()}/{len(y_test)})")

# Check positive class proportion by year
print("\nTraining set positive class proportion by year:")
train_year_stats = train_data.groupby('year')['sp500_drop_8pct_next_3m'].agg(['sum', 'count', 'mean'])
for year in sorted(train_year_stats.index):
    stats = train_year_stats.loc[year]
    print(f"  {year}: {stats['mean']:.2%} ({int(stats['sum'])}/{int(stats['count'])})")

print("\nTest set positive class proportion by year:")
test_year_stats = test_data.groupby('year')['sp500_drop_8pct_next_3m'].agg(['sum', 'count', 'mean'])
for year in sorted(test_year_stats.index):
    stats = test_year_stats.loc[year]
    print(f"  {year}: {stats['mean']:.2%} ({int(stats['sum'])}/{int(stats['count'])})")

# 4. Feature Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Handle Class Imbalance
positive_count = y_train.sum()
negative_count = len(y_train) - positive_count
scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1

print(f"\nClass imbalance handling - scale_pos_weight: {scale_pos_weight:.2f}")

# 6. Train XGBoost Model
print("\nTraining XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=20
)

xgb_model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

# 7. Predictions and Evaluation
print("\nEvaluating model performance...")
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0

print("\n" + "=" * 60)
print("Model Performance Evaluation on 2015-2019 Test Set")
print("=" * 60)
print(f"Accuracy:         {accuracy:.4f}")
print(f"Precision:        {precision:.4f}")
print(f"Recall:           {recall:.4f}")
print(f"F1 Score:         {f1:.4f}")
print(f"AUC Score:        {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Drop', 'Will Drop']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Predicted: No Drop', 'Predicted: Will Drop'],
    yticklabels=['Actual: No Drop', 'Actual: Will Drop']
)
plt.title('Confusion Matrix - 2015-2019 Predictions')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# 8. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("Feature Importance Analysis")
print("=" * 60)
print("Top 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_n = min(15, len(feature_importance))
top_features = feature_importance.head(top_n)

plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title(f'XGBoost Feature Importance (Top {top_n})')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 9. Save Model and Related Objects
print("\nSaving model and related objects...")
model_info = {
    'model': xgb_model,
    'scaler': scaler,
    'vectorizer': vectorizer if 'vectorizer' in locals() else None,
    'numeric_features': numeric_features,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': auc,
    'train_years': '2008-2014',
    'test_years': '2015-2019',
    'feature_names': X_train.columns.tolist()
}

joblib.dump(model_info, 'xgb_sp500_2008_2014_model.pkl')
print("Model saved as 'xgb_sp500_2008_2014_model.pkl'")

# 10. Yearly and Monthly Analysis of Predictions
print("\n" + "=" * 60)
print("Yearly and Monthly Prediction Results Analysis")
print("=" * 60)

# Add predictions to test data
test_data_with_pred = test_data.copy()
test_data_with_pred['prediction'] = y_pred
test_data_with_pred['prediction_probability'] = y_pred_proba
test_data_with_pred['prediction_correct'] = (
    test_data_with_pred['prediction'] == test_data_with_pred['sp500_drop_8pct_next_3m']
)

# Yearly statistics
test_data_with_pred['year'] = test_data_with_pred['date'].dt.year
yearly_stats = test_data_with_pred.groupby('year').agg({
    'sp500_drop_8pct_next_3m': 'sum',
    'prediction': 'sum',
    'prediction_correct': 'mean',
    'prediction_probability': 'mean'
}).rename(columns={
    'sp500_drop_8pct_next_3m': 'actual_drop_count',
    'prediction': 'predicted_drop_count',
    'prediction_correct': 'accuracy',
    'prediction_probability': 'avg_prediction_probability'
})

print("Yearly Statistics:")
print(yearly_stats)

# Monthly statistics
test_data_with_pred['month'] = test_data_with_pred['date'].dt.month
monthly_stats = test_data_with_pred.groupby('month').agg({
    'sp500_drop_8pct_next_3m': 'sum',
    'prediction': 'sum',
    'prediction_correct': 'mean',
    'prediction_probability': 'mean'
}).rename(columns={
    'sp500_drop_8pct_next_3m': 'actual_drop_count',
    'prediction': 'predicted_drop_count',
    'prediction_correct': 'accuracy',
    'prediction_probability': 'avg_prediction_probability'
})

print("\nMonthly Statistics:")
print(monthly_stats)

# Visualize yearly and monthly accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(yearly_stats.index, yearly_stats['accuracy'], alpha=0.7)
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy: {accuracy:.2f}')
plt.xlabel('Year')
plt.ylabel('Accuracy')
plt.title('2015-2019 Yearly Prediction Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(monthly_stats.index, monthly_stats['accuracy'], alpha=0.7)
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy: {accuracy:.2f}')
plt.xlabel('Month')
plt.ylabel('Accuracy')
plt.title('2015-2019 Monthly Prediction Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize positive class proportion over time
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Training set positive class proportion
train_yearly_pos_rate = train_year_stats['mean']
axes[0].bar(train_yearly_pos_rate.index, train_yearly_pos_rate.values, alpha=0.7)
axes[0].axhline(
    y=y_train.mean(), color='r', linestyle='--',
    label=f'Training Avg: {y_train.mean():.2%}'
)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Positive Class Proportion')
axes[0].set_title('Training Set (2008-2014) - Positive Class Proportion by Year')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test set positive class proportion
test_yearly_pos_rate = test_year_stats['mean']
axes[1].bar(test_yearly_pos_rate.index, test_yearly_pos_rate.values, alpha=0.7)
axes[1].axhline(
    y=y_test.mean(), color='r', linestyle='--',
    label=f'Test Avg: {y_test.mean():.2%}'
)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Positive Class Proportion')
axes[1].set_title('Test Set (2015-2019) - Positive Class Proportion by Year')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("Model Training and Evaluation Complete!")
print("=" * 60)
print(f"Trained on 2008-2014 data ({len(train_data)} samples)")
print(f"Tested on 2015-2019 data ({len(test_data)} samples)")
print(f"Final Accuracy: {accuracy:.2%}")
print(f"Final F1 Score: {f1:.4f}")
print("Model saved and ready for predicting new data")
```

### 4.4 Reflections and Limitations

This modeling stage reinforced several lessons:

Time-based evaluation is stricter but more honest. Performance usually drops compared to random split, but the result is more credible.

Rare-event prediction is inherently hard. A high accuracy can be misleading if the model simply predicts “no crash” most of the time.

Interpretability matters. Even when performance is not perfect, understanding which text signals drive predictions is valuable for building a defensible warning framework.

Overall, this step helped us close the loop from “text signals” to “actionable warning indicators,” while staying cautious about over-claiming predictive power.


## 5. Modeling: SVM

After the modeling stage with XGBoost, we achieved a good performance, especially excelling in handling high-dimensional and non-linear features. However, as an ensemble learning method, XGBoost, while powerful in prediction, can sometimes be prone to overfitting and has longer training times. To gain a more comprehensive evaluation of the model performance, we decided to benchmark Support Vector Machine (SVM) as a baseline model.

SVM is a classic binary classification model that performs particularly well with high-dimensional data and excels in some simple linear classification tasks. Compared to XGBoost, SVM requires fewer hyperparameter adjustments during training and is more computationally efficient. Therefore, when computational resources are limited or rapid prototyping is needed, SVM is a good choice. By comparing the performance of XGBoost and SVM, we can better understand the strengths and limitations of each model when handling our dataset, helping us make a more informed model selection.

### 5.1. XGBoost vs SVM: Benchmark Comparison

**To contextualize the performance of XGBoost, we benchmark it against a Support
Vector Machine (SVM)**, a widely used and well-established classification model
that serves as a strong baseline in financial prediction tasks. Both models are
trained and evaluated on identical datasets using the **same time-based
train–test split and consistent preprocessing procedures**.

##### 5.1.1 SVM: Core Principles

Support Vector Machine is a powerful supervised learning algorithm originally
designed for binary classification. Its core idea is to find an **optimal**
hyperplane in the feature space that maximizes the margin between two classes.

The SVM optimization objective can be expressed as:

$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w \cdot x_i + b) \geq 1, \forall i$$

where $w$ is the normal vector of the hyperplane, $b$ is the bias term, and
$y_i \in \{-1, +1\}$ represents the class labels.

For data that is not linearly separable, SVM uses the **Kernel Trick** to map
original features into a higher-dimensional space where the data becomes
linearly separable. Common kernel functions include **Linear**, **RBF**,
**Polynomial** and **Sigmoid**.

### 5.1.2 SVM: Experimental Design

##### 5.1.2.1 Data Preprocessing

Consistent with the XGBoost model, all features were processed using the same
pipeline to eliminate the effect of differing feature scales.

#### 5.1.2.2 Kernel Selection Experiment

To select the optimal kernel function, we performed **GridSearch**
hyperparameter tuning for all four kernels. The search space included
regularization parameter C for all kernels, with kernel-specific parameters
(gamma for RBF/sigmoid, degree for polynomial). For example, the RBF parameter
grid:

```python
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Parameter grid for RBF kernel
param_grid = {
    'C': [0.1, 1, 10, 50, 100, 500],
    'gamma': [0.0005, 0.001, 0.01, 0.1]
}
```

We used **TimeSeriesSplit** for cross-validation (3 folds) to ensure the
training process respects temporal ordering and prevents data leakage. The
implementation is as follows: 

```python
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

kernels = ['linear', 'rbf', 'poly', 'sigmoid']

for k in kernels:
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=SVC(kernel=k, class_weight='balanced', probability=True, random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=tscv,
        n_jobs=-2,
        verbose=0
    )
    grid_search.fit(X_train_scaled, y_train)
    svm_model = grid_search.best_estimator_
```


Following **GridSearch**
optimization, the optimal performance achieved by each kernel is presented
below:

![](./images/graph1.png)

### 5.1.3 SVM: Why We Chose the RBF Kernel

After systematically comparing all four kernel functions, we selected the
**RBF** kernel for our final model. The key reasons are as follows:

#### 5.1.3.1 Non-linear Modeling Capability

Financial markets operate under complex dynamics, where price movements often
exhibit **non-linear relationships** with various factors. The RBF kernel maps
data into an infinite-dimensional feature space, enabling it to capture these
complex non-linear patterns effectively.

#### 5.1.3.2 Robustness and Generalization

The RBF kernel possesses a **locality property**. This characteristic makes the
model: **(1)** more robust to noisy data, **(2)** less prone to overfitting to
extreme values in the training set **and (3)** more stable on unseen test data.

#### 5.1.3.3 Parameter Tunability

The RBF kernel has two key hyperparameters:

- **C (Regularization parameter)**: Controls the trade-off between
  classification errors and margin size
- **γ (Gamma)**: Controls the influence range of individual samples

Compared to the polynomial kernel, which requires tuning the degree parameter,
the RBF kernel's parameter space is easier to search and shows relatively lower
sensitivity to parameter changes.

#### 5.1.3.4 Why Not Linear Kernel

Although the linear kernel achieved performance comparable to the RBF kernel, we
ultimately selected the RBF kernel. Given the inherently non-linear nature of
financial market dynamics, the flexibility of the RBF kernel allows it to better
capture complex relationships and adapt to potential regime changes. Therefore,
in the presence of similar empirical performance, the RBF kernel was preferred
for its stronger theoretical suitability and expected robustness in
out-of-sample settings.

### 5.1.4 SVM: Model Performance

#### 5.1.4.1 Evaluation Metrics

We evaluated model performance on the 2015–2019 test set using the metrics same
as XGBoost, including **Accuracy**, **Precision**, **Recall**, **F1-Score** and
**AUC**.

#### 5.1.4.2 Confusion Matrix Analysis

Under the **RBF kernel with the optimal hyperparameter** configuration, the
confusion matrix illustrates the model’s classification behavior across
different outcome categories. The **True Positive Rate** and **False Positive
Rate** are as below:

![](./images/graph2.png)

#### 5.1.4.3 Threshold Tuning

We also plotted the **Precision-Recall Curve** and **Performance threshold** to
analyze model performance under different thresholds, providing guidance for
threshold selection in practical applications.

![](./images/graph3.png)

![](./images/graph4.png)

### 5.1.5 XGBoost vs SVM: Model Comparison

We conducted a comprehensive comparison across the following dimensions:

1. **Overall Metrics Comparison**: As shown below, XGBoost achieves higher
   scores across all three key metrics: Accuracy, F1, and AUC.

![](./images/graph5.png)

2. **Superior ROC Curve**: The ROC curve comparison reveals that XGBoost (AUC =
   0.907) significantly outperforms SVM (AUC = 0.856), demonstrating better
   discrimination ability between positive and negative samples across all
   classification thresholds.

![](./images/graph6.png)

3. **Better Precision-Recall Trade-off**: XGBoost achieves substantially higher
   Average Precision (AP = 0.525) compared to SVM (AP = 0.329), indicating
   stronger performance in handling class imbalance.

![](./images/graph7.png)

4. **Confusion Matrix Analysis**: Both models show high accuracy on the majority
   class ("No Drop"), with XGBoost achieving 96.85% versus SVM's 93.62%. For the
   minority class ("Drop"), both models exhibit similar recall rates
   (approximately 52-53%), reflecting the inherent challenge of predicting
   market drops.

![](./images/graph8.png)

### 5.1.6 Conclusion

This benchmark comparison demonstrates that **XGBoost outperforms SVM on most
metrics** for S&P 500 market drop prediction. While SVM serves as a
well-established baseline, our comprehensive evaluation reveals clear
performance gaps across key metrics:

| Metric    | XGBoost | SVM (RBF) |
| --------- | ------- | --------- |
| Accuracy  | 0.9448  | 0.9148    |
| AUC       | 0.9069  | 0.8562    |
| Precision | 0.4778  | 0.3158    |
| Recall    | 0.5181  | 0.5301    |
| F1        | 0.4971  | 0.3958    |

XGBoost's dominance stems from its **ensemble learning architecture**, which
automatically captures complex feature interactions and handles class imbalance
more effectively than SVM's implicit kernel mapping.
