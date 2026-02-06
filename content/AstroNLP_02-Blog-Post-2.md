---
Title: Blog Post 2 (by Group "AstroNLP")
Date: 2026-01-17 01:12
Category: Reflective Report
Tags: Group AstroNLP
---

By Group "AstroNLP"
> >*The analysis shown in the blog is strictly from a financial and market impact perspective.*

In Blog Post 1, we documented the practical realities of collecting and cleaning financial news data. In this second post, we focus on a different angle: **how much useful structure we can extract from short news titles using K-means**, and whether those text-derived clusters show*any measurable linkage with next-day gold price movements—especially under different volatility regimes. We work with a dataset of gold-related financial news headlines collected from publicly available financial news sources and matched with daily gold price movements.

A key lesson we learned is that *getting results is easy; knowing which results are useful is the hard part*. Therefore, this post is deliberately **numbers-first** and **judgment-heavy**: we explain what is informative, what is not, and what we will change next.

---

## 1. Why clustering headlines is harder than it looks

We cluster **gold-related news titles** (not full articles). This is important because titles are short, compressed, and often reuse overlapping phrases (e.g., “gold prices”, “settles”, “dollar”, “ounce”). That design choice is realistic (titles are widely available), but it comes with a cost:

- The vocabulary overlap is high.
- Many titles are “template-like” (e.g., end-of-day summaries).
- TF-IDF vectors built from short texts are sparse and noisy.

So our goal is **not** to produce perfectly separated clusters in the geometric sense. Instead, we aim to get **economically interpretable partitions** that can serve as an information layer for market linkage analysis.

---

## 2. Choosing K: silhouette scores are low

We evaluated K from 2 to 8 using a subsample for speed. The code snippet below is reproducible and small enough for a blog context:

```python
def silhouette_elbow(X, Ks=range(2, 9), sample_size=2000):
    n = min(sample_size, X.shape[0])
    idx = np.random.RandomState(42).choice(X.shape[0], size=n, replace=False)
    X_small = X[idx] 

    scores = []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_small)
        scores.append(silhouette_score(X_small, labels))
    return scores

if __name__ == "__main__":
    X = load_tfidf()
    Ks = range(2, 9)
    scores = silhouette_elbow(X, Ks)
    print(list(zip(Ks, scores)))
```

Our output (headline TF-IDF) was:

```
[(2, np.float64(0.012574001577492958)), (3, np.float64(0.017861588261286507)), (4, np.float64(0.021068812973244085)), (5, np.float64(0.022575536810476866)), (6, np.float64(0.021716745657916026)), (7, np.float64(0.023710576644436997)), (8, np.float64(0.023979766133093612))]
```

**Interpretation (what this means and why it matters):**

- All silhouette scores are below 0.03 → weak cluster separation in TF-IDF space.
- This is expected for short titles with shared vocabulary.
- The low silhouette scores are a warning: clusters should not be treated as “clean labels” or “hard truth”.

Still, choosing K=4 gives a reasonable trade-off between interpretability and fragmentation, and it matches the thematic patterns we observed in keyword inspection.

---

## 3. What the clusters actually represent

We used TF-IDF with unigrams and bigrams and applied K-means with K=4:

```python
# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2
)
X = vectorizer.fit_transform(df["clean_news"])

# KMeans
K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)
```

the output shows:
```
Cluster 0 top terms:
gold ends, ends, ounce, gold ounce, gold, nymex, ounce nymex, higher, ends lower, ends higher, lower, ends ounce

Cluster 1 top terms:
oz, dec gold, gold oz, dec, gold, settle oz, settle, settles oz, june gold, gold settles, settles, june

Cluster 2 top terms:
gold futures, futures, gold, rs, ounce, futures ounce, cues, close, week, global cues, global, futures close

Cluster 3 top terms:
gold, prices, gold prices, silver, dollar, rs, gold silver, gains, demand, week, trade, high
                                          news_title  cluster
0  april gold down 20 cents to settle at $1,116.1...        1
1          gold suffers third straight daily decline        3
2     Gold futures edge up after two-session decline        2
3  dent research : is gold's day in the sun comin...        3
4  Gold snaps three-day rally as Trump, lawmakers...        3
```

Based on top TF-IDF terms, we interpret clusters as:

- Cluster 0 (low information): “ends / settles”
    - Mostly backward-looking descriptive titles (e.g., “gold ends lower”).
    - **Usefulness:** low. These titles reflect what already happened.

- Cluster 1 (low-to-medium information): “contract month / settlement mechanics”
    - Often technical market microstructure (delivery months, settlement references).
    - **Usefulness:** limited. More about contract labeling than fundamentals.

- Cluster 2 (medium information): “global cues / broader financial conditions”
    - Captures cross-market / risk sentiment cues.
    - **Usefulness:** meaningful as a macro-sentiment channel.

- Cluster 3 (higher information): “dollar / demand / silver / macro drivers”
    - Macro and cross-asset drivers appear frequently.
    - **Usefulness:** higher, because these relate to expectation-setting and risk pricing.

**Cluster imbalance (an important diagnostic)**

Our cluster distribution is highly uneven:

- Cluster 3: 68.79%
- Cluster 2: 15.18%
- Cluster 1: 10.46%
- Cluster 0: 5.58%

This explains two things:

- why silhouette scores are low (a dominant “macro-like” cluster absorbs many points), and
- why clustering on short titles is not sufficient as a standalone trading signal generator.

In short: clustering is helpful as an information organizer, not as a magic alpha machine.

---

## 4. Market linkage: do clusters correlate with next-day returns?

We merge cluster labels with gold next-day return and build a direction label `up`:

```python
from scipy.stats import chi2_contingency, ttest_1samp

gold = pd.read_csv("gold_price_cleaned.csv")
gold["date"] = pd.to_datetime(gold["date"])
gold = gold.sort_values("date")
gold["return"] = gold["price"].pct_change()
gold["next_day_return"] = gold["return"].shift(-1)

news_df["news_date"] = pd.to_datetime(news_df["news_date"])
merged = news_df.merge(
    gold[["date", "next_day_return"]],
    left_on="news_date",
    right_on="date",
    how="left"
)

merged["up"] = (merged["next_day_return"] > 0).astype(int)
```

### 4.1 Average next-day return by cluster

Mean next-day returns by cluster were:

```
Cluster 0: 0.000115

Cluster 1: 0.000226

Cluster 2: 0.000253

Cluster 3: 0.000358
```

This gives a ranking (3 > 2 > 1 > 0), consistent with the “information density” interpretation. However, ranking is not the same as statistically reliable alpha.

---

### 4.2 Directional linkage: chi-square test says “weak but detectable”

We tested whether `cluster` and `up` are independent:

```python
contingency_table = pd.crosstab(merged['cluster'], merged['up'])
chi2, p_chi2, _, _ = chi2_contingency(contingency_table)

print("\nChi-Square Test:")
print(f"Chi-Square Statistic: {chi2}, p-value: {p_chi2}")
```
Output：
```
Chi-Square Test:
Chi-Square Statistic: 8.53737682701176, p-value: 0.03611808803010272
```

Our p-value was **0.036**, which rejects independence at the 5% level. This is one of the most meaningful results so far:

- It suggests cluster membership carries some information about direction, even if weak.
- It supports the idea that clustering captures a real structure in the text—not just noise.

---

### 4.3 But: return magnitude is not significant (t-tests fail)

When we compare each cluster’s mean return to the overall baseline using t-tests, all p-values were large (not significant). In other words:

- Cluster membership alone does not generate statistically significant excess returns.

**So clusters are not a direct trading rule by themselves.**

This distinction is critical: clusters can be informative without being directly tradable.

---

## 5. Volatility regimes: where the signal becomes more meaningful

We introduce a volatility regime variable (high vs low) using the median of the gold volatility index:

```python
vol_regime_stats = merged.groupby(['vol_regime', 'cluster']).agg(
    mean_return=('next_day_return', 'mean'),
    upward_probability=('up', 'mean')
).reset_index()

print("\nMean Next-Day Return by Volatility Regime and Cluster:")
print(vol_regime_stats)
```
Output：
```
Mean Next-Day Return by Volatility Regime and Cluster:
   vol_regime  cluster  mean_return  upward_probability
0         0.0        0    -0.000070            0.463768
1         0.0        1    -0.000051            0.514821
2         0.0        2    -0.000344            0.504587
3         0.0        3     0.000090            0.506768
4         1.0        0     0.000612            0.512097
5         1.0        1    -0.000006            0.497537
6         1.0        2     0.001509            0.525912
7         1.0        3     0.000676            0.501615
```
**Conditional mean returns (key finding)**

Mean next-day returns by regime and cluster:

- **Low volatility (vol_regime=0):**
    - C0: -0.000070
    - C1: -0.000051
    - C2: -0.000344
    - C3: 0.000090

- **High volatility (vol_regime=1):**
    - C0: 0.000612
    - C1: -0.000006
    - C2: 0.001509
    - C3: 0.000676

**Interpretation:**

- Under low volatility, cluster differences are small or even negative.
- Under high volatility, macro-related clusters (2 and 3) show larger positive next-day returns, especially Cluster 2 (+0.1509%).

This appears to be a relatively informative conditional pattern:

- Clusters become more informative when the market is already uncertain.

At the same time, we remain cautious:

- Even 0.15% is not necessarily tradable after costs.
- We treat this as informational linkage, not guaranteed profit.

---

## 6. Digging Deeper: Relative Upward Probabilities and Returns 

To better interpret the economic importance of clusters, we computed **relative upward probability** and **relative return** for each cluster. These metrics compare each cluster’s performance against the overall baseline expectations.

```python
cluster_stats["relative_upward_probability"] = cluster_stats["upward_probability"] - overall_upward_probability
cluster_stats["relative_return"] = cluster_stats["avg_return"] - overall_avg_return
```

- **Relative Upward Probability:** Measures whether the chance of next-day "upward movement" (return > 0) is above or below the global baseline. Positive values suggest the cluster captures signals leading to upward trends with higher probability than the baseline.
  \[
  \text{Relative Upward Probability} = \text{Cluster Upward Probability} - \text{Overall Upward Probability}
  \]
- **Relative Return:** Measures whether the average next-day return of a cluster is higher or lower than the global average return. Positive values suggest the cluster potentially generates higher returns relative to the baseline.
  \[
  \text{Relative Return} = \text{Cluster Average Return} - \text{Overall Average Return}
  \]

```
   cluster  sample_size  ...  relative_upward_probability  relative_return
0        0          589  ...                     0.010496        -0.000197
1        1         1105  ...                     0.029336        -0.000086
2        2         1603  ...                     0.017650        -0.000059
3        3         7266  ...                    -0.009206         0.000045
```
**Interpretation:**
1. **Cluster 1 (Technical/Settlement News):** 
   - Exhibits the **highest relative upward probability** at +0.0293, meaning that news titles falling in this cluster are more likely to correlate with upward next-day market moves compared to the overall dataset.
   - However, relative return is still slightly negative, indicating that this information may not translate into meaningful price increases after considering the baseline.

2. **Cluster 2 (Global Cues/Financial Sentiment):**
   - Shows a strong **relative upward probability** (+0.0177), suggesting that risk sentiment and cross-market cues are informative about directionality.
   - Relative returns are less negative than Cluster 0 and Cluster 1, making these signals more economically relevant.

3. **Cluster 3 (Macro Drivers):**
   - While it dominates in terms of sample size (7,266 samples, 68% of the dataset), Cluster 3 demonstrates a **negative relative upward probability** (-0.0092), indicating a slightly lower likelihood of upward moves.
   - However, **positive relative returns** suggest macro trends may still hint at minor risk-adjusted profitability, especially under certain filter conditions.

4. **Cluster 0 (Low-Information Summaries):**
   - As expected, this cluster adds little actionable value:
     - Relative upward probability is small (+0.0105), reflecting little improvement over randomness.
     - Returns are also weak or slightly negative.

In summary:
- Clusters 1 and 2 signal potential economic value, but **Cluster 2’s global/macro content** appears slightly more promising overall.
- Cluster 3, while containing the dominant share of data, shows mixed signals; its potential utility likely depends on volatility regimes or additional modeled filters.
- Cluster 0 confirms that backward-looking, low-information summaries are **not useful predictors**.

**Implications**
While relative metrics offer meaningful insights, they also expose the limitations of clustering:
- Even “standout” clusters have relative effects that are **small in magnitude**. For example, a +0.03 increase in upward probability is unlikely to generate immediate gains after incorporating transaction costs.
- Clustering should therefore be treated as a **signal refinement step**, not an independent trading signal generator.
---

## 7. Conclusion and Reflection
 Working with short financial news headlines and unsupervised clustering taught us an important lesson: extracting structure from text is much easier than extracting economically meaningful signals. While modern NLP tools make it straightforward to produce clusters and statistical results, interpreting whether those results carry genuine informational value requires careful judgment and restraint.

Throughout this project, we learned that clustering performance metrics alone—such as silhouette scores—can be misleading when applied to short, highly repetitive texts. In our case, low silhouette values did not indicate failure, but rather reflected the inherent limitations of headline-level data. This reinforced the idea that interpretability and economic reasoning matter more than purely geometric separation in financial text analysis.

The key takeaways from this stage of the project are:

- Short news titles contain limited but non-random structure, even when traditional clustering metrics suggest weak separation.
- Cluster membership can capture directional information, but does not directly translate into statistically significant excess returns.
- Context matters: the informational value of text-derived clusters becomes more visible under high-volatility regimes.
- Clustering is best viewed as an information-organizing layer, not a standalone trading signal generator.
- Statistical detectability is not the same as tradability, especially after accounting for transaction costs and market noise.

Equally important were the conceptual lessons about model evaluation. This project reminded us that producing numerical outputs is only the starting point; the harder task is deciding which results are robust, which are fragile, and which should be discarded entirely. In several instances, results that initially appeared promising lost significance under closer inspection, highlighting the importance of skepticism in empirical financial research.

Looking ahead, this experience will shape how we approach the next stages of the project. Rather than relying on headline clustering alone, we plan to treat textual clusters as conditional signals—to be combined with volatility filters, richer text representations, and complementary market features. Overall, this blog post documents not just what we built, but how our understanding of the limits and proper use of NLP in finance has evolved.
