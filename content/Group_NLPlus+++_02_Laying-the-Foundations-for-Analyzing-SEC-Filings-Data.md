---
Title: Laying the Foundations for Analyzing SEC Filings Data
Date: 2026-01-21 13:00
Category: Reflective Report
Tags: Group NLPlus+++, Literature Review, Web Scraping
---


## Blog 2

## Introduction

In our previous post, we discussed the nuances of preprocessing earnings calls. However, as menioned in our presentations, due to no Capital IQ API access provided by the University, along with the Bloomberg Terminal setting a daily download limit, we have pivoted our data source to focus sole on 10K and 10Q filings, which are readily available to be scraped from the SEC. However, before writing a single line of code, we had to go through multiple sources of literature to gain inspiration.

Our project is not just about downloading data; it is about replicating and extending interesting NLP research to find predictive trading signals in financial text. In this post, we first outline the three academic pillars inspiring our strategy, and then dive into the technical engineering required to build the pipeline that supports them. It requires handling strict government rate limits, choosing the right data structures for efficiency, and selecting parsers that don't crash on messy legacy HTML.

## Part 1: Literature Review - Inspirations for our Research Direction

Our research direction is grounded in three key studies that guide how we assess risk, value narratives, and summarize vast amounts of text.

### 1. Beyond the 10-K: Generalizing Risk Quantification

Our framework for assessing textual risk starts with **Grundy and Petry’s (2020)** study, *“10-K Risk Factors Quantification and the Information Content of Textual Reporting.”* They demonstrated that qualitative risk narratives in 10-K filings can be systematically quantified using machine learning (specifically LDA topic modeling). Crucially, these textual measures explain a substantial portion of cross-sectional variation in observable firm risk—proving that even “boilerplate” disclosures contain signal.

Grundy and Petry found that 10-K disclosures are largely contemporaneous, exhibiting limited forward-looking power. Building on this, we originally planned to generalize their framework to multiple disclosure channels, including 10-Q reports, analyst research reports, and earnings call transcripts, which differ in frequency and incentives. However, we have settled on 10K/10Q only due to the aforementioned issues regarding data constrnaints. Though one of the paper's conclusions was that 10K text contains little forward-looking predictive power, we aim to assess this ourselves, and merge them with other NLP preprocessing/feature engineering approaches to examine whether combined textual indicators can unlock forward-looking predictive capabilities in both predicting returns and realized volatility.

### 2. Mining Value in Analyst Reports

For our trading strategy, we draw inspiration from the study *“Do Sell-side Analyst Reports Have Investment Value?”* (2025). This research utilizes Large Language Models (LLMs) and Shapley value decomposition to extract investment value from qualitative narratives.

The study found that a long-short strategy based on text-derived predictions generated significant alpha (1.04% monthly). Interestingly, this predictability often stems from reports featuring short-term negative sentiment but long-term optimistic fundamentals, which is a pattern reflecting market overreaction to near-term bad news. Furthermore, the “Strategic Outlook” sections of these reports were found to drive a massive 41% of the portfolio's Sharpe ratio. If we could replicate this “Strategic Outlook” filtering to isolate high-value signals from noise, then we could use LLM-derived embeddings to capture the nuance between temporary setbacks and fundamental decline. However, as we found difficulties with obtaining a rich dataset of many sell-side analyst reports, we focused on the 10-K paper instead.

### 3. Taming Long Documents with ECTSum

Finally, handling the sheer volume of texts may require advanced summarization techniques. We looked to adopt the framework from Mukherjee et al.’s *“ECTSum: A New Benchmark Dataset for Bullet Point Summarization of Long Earnings Call Transcripts.”*

ECTSum, the first large-scale financial long-document summarization dataset, comprises 2,425 pairs of unstructured earnings call transcripts (average 2.9K words) and expert-written telegram-style bullet-point summaries, with an ultra-high compression ratio of 103.67. The ECT-BPS framework they developed integrates a FinBERT-enhanced extractive module with a T5-based paraphrasing module. It has been shown to outperform state-of-the-art models in ROUGE, BERTScore, factual consistency metrics, and financial expert evaluations. Notably, both the dataset and ECT-BPS are publicly available on GitHub. Building on earnings call transcripts, we aim to explore ECT-BPS’s application potential in 10-K and 10-Q filings, to investigate whether key financial information extracted via ECTSum can unlock novel predictive capabilities for financial analysis. Should using the ECTSum methodology or training our own model prove too computationally expensive, running pre-trained or open-source summarizers such as BART are also viable.

## Part 2: Engineering the Pipeline

With our research directions mostly set, the next challenge was execution. Building a system to analyze many years of SEC filings isn't just about writing a script that works once only, but about building a reproducible, robust data engine with structured outputs.

We redesigned our workflow into a three-stage pipeline: **Universe Construction → High-Concurrency Scraping → Surgical Extraction.**

### 1. Defining the Universe: Quality Over Quantity

**Corresponding Workflow:** `wrds_data` → `build_cik_universe`

Before downloading a single PDF, we had to answer: Which companies matter?

A naive approach would be to download filings for every company listed on the SEC. However, this introduces noise (shell companies, penny stocks) and ignores economic reality. Furthermore, simply using today's S&P 500 constituents to backtest data from 2010 introduces Survivorship Bias, i.e. we would miss companies like Kodak or Lehman Brothers that were significant then but are gone now.

**Our Solution: Dynamic Annual Rebalancing**

We built a “Universe Construction” module that ingests raw market data (from WRDS/CRSP).

- **Market Cap Ranking:** For every year (e.g., June 2010, June 2011...), we rank all public firms by market capitalization
- **Top N Selection:** We dynamically select the Top 100 firms for each year
- **Identifier Mapping:** We map financial identifiers (GVKEY/PERMNO) to SEC identifiers (CIK) to ensure our financial data aligns perfectly with the text data

This ensures our model trains on the most economically significant companies at that point in time.

### 2. The Scraper: Rate Limiting & Concurrency

**Corresponding Workflow:** `sec_scrape_filings`

The SEC's EDGAR database is a goldmine of textual data, but online sources cited that it enforces a rate limit, resulting in a risk of an IP ban if it is exceeded. A simple single-threaded loop is too slow for thousands of files, but uncontrolled multi-threading triggers bans.

**Our Solution: The Token Bucket Algorithm**

Instead of simple `time.sleep()`, we implemented a Token Bucket rate limiter.

- **Concept:** Imagine a bucket that fills with “tokens” at a rate of 10 per second. Every download thread must “pay” a token to send a request. If the bucket is empty, the thread waits.
- **Concurrency:** This allowed us to wrap our scraper in a `ThreadPoolExecutor`, launching multiple download workers simultaneously.
- **Safety:** The global Token Bucket ensures that even with 10 parallel threads, the aggregate traffic never violates SEC rules.

This architecture maximizes throughput while maintaining 100% compliance with government API limits.

### 3. Surgical Extraction: Parsing “Tag Soup” with lxml

Financial filings are notoriously messy. While modern HTML is structured, older 10-K filings often contain broken tags, unclosed elements, and inconsistent formatting (often called “tag soup”). A risk model fed with the “Table of Contents” or “Page Headers” will fail.

**Targeted Regex Extraction**

We developed a library of Regular Expressions to identify the specific boundaries of key sections in both 10-K (Annual) and 10-Q (Quarterly) reports:

- **Item 1A (Risk Factors):** Captures the company's self-assessed risks.
- **Item 7 (MD&A):** Captures management's narrative on performance.
- **Item 3 (Legal Proceedings):** Captures litigation risks.

To extract text from these files, we used the BeautifulSoup library. However, BeautifulSoup is just a wrapper; it requires an underlying parser to do the heavy lifting. We specifically chose `lxml` over Python's built-in `html.parser`.

**Why `lxml`?**

1. **Speed:** `lxml` is written in C, making it significantly faster at parsing large HTML trees than Python's built-in tools.
2. **Lenience & Stability:** Financial data is often messy. `lxml` is extremely robust at handling broken HTML without crashing or throwing errors. It attempts to “fix” the structure as it reads, ensuring we extract usable text even from poorly formatted filings.

**Implementation:**

In our `extract_text_from_html` function, you will see this specific argument:

```python
from bs4 import BeautifulSoup

def extract_text_from_html(html_content):
    # 'lxml' is explicitly defined as the parser
    soup = BeautifulSoup(html_content, 'lxml')

    # Clean up scripts and styles
    for element in soup(['script', 'style']):
        element.decompose()

    return soup.get_text(separator=' ')
```

### 4. Micro-Optimization: Why Set Beats List

When processing thousands of financial documents, “small” inefficiencies compound into hours of wasted compute time. One optimization we implemented in our preprocessing code (shared previously) is using Sets (`set`) instead of Lists (`list`) for tasks like stopword removal.

**Why: O(.) complexity

- **List Lookup:** `O(n)` complexity. Checking if a word exists in a list (e.g., `if word in stop_words_list`) requires scanning the whole list until a match is found.
- **Set Lookup:** `O(1)` complexity. Hash tables make lookups instant on average.

**The Impact:**

A standard stopword list might contain ~180 words. A 10-K filing can easily contain 50,000+ words.

1. **Using a List:** 50,000 words × 180 checks = ~9,000,000 operations per document.
2. **Using a Set:** 50,000 words × 1 check = ~50,000 operations per document.

In our code, this looks like:

```python
# Inefficient
stop_words_list = ["the", "is", "at", ...]

# Efficient
STOP_WORDS = set(stopwords.words('english'))

# usage
processed_words = [word for word in words if word not in STOP_WORDS]
```

This simple change drastically reduced our preprocessing time, allowing us to iterate on our LDA models much faster.

## Conclusion

Data engineering often goes unnoticed in the final results and findings of trading strategies and predictions, but it is the backbone of our project's evaluation of code quality. By iterating and moving from a simple script to a  pipeline that declares a stock universe, limites scraping rates, and downloads specific 10-K/10-Q sections in a robust manner, we have built a resilient foundation capable of handling the scale required for our research.
