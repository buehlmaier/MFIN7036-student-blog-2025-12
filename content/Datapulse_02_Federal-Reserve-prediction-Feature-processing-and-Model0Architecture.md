---
Title: Federal Reserve prediction - Feature processing and Model Architecture  (by Group "DataPulse")
Date: 2026-01-21 23:00
Category: Reflective Report
Tags: Group DataPulse
---

This is the second blog post. 

In our previous article, we detailed the systematic approach to collecting and preprocessing FOMC meeting statements and minutes. Data cleaning, standardization, and structuring form the foundation for building reliable analytical models. Today, we will further explore how to extract meaningful features from these standardized, highly policy-oriented texts and develop predictive models to identify the Federal Reserve's monetary policy signals.


##Fed Text Purification Practice: Problems, Reflections, and Solutions##

In the last version, our basic text processing (just stripping punctuation, basic stop words, etc.) turned out to be nowhere near enough for real analysis. Noise completely took over, models kept getting distracted by all the repetitive, low-information phrases, and the subtle shifts in the Fed’s policy tone and signals were basically lost. That made me stop and rethink: the issue wasn’t that we hadn’t cleaned “enough,” but that we hadn’t really understood what each document type is actually trying to do and what information really matters in it. So this time we built two very different purification approaches — one for **Minutes** (the long meeting minutes) and one for **Statement** (the short policy announcements).

##Minutes: Why Do All Those Procedural Boilerplate Phrases Cause So Much Trouble?##

**Problem** Minutes are usually extremely long and full of structural boilerplate — things like staff review opening lines, manager reports, liquidity facility descriptions, voting procedures, notation votes, and so on. These parts are highly templated and repeat themselves over and over, but they add almost nothing to understanding real economic views or policy disagreements. In my early experiments, word clouds and top TF-IDF terms were almost entirely flooded with phrases like “the Committee agreed” or “staff judged.” The actual substantive discussion got buried, and stance classification accuracy stayed disappointingly low.

**Reflection** At the beginning I thought these “official-sounding” parts were necessary for the document’s formality. But after reading and comparing many different Minutes releases side by side, I realized they’re basically just fixed scaffolding for meeting records — kind of like the procedural “minutes of the meeting” formalities you see in any board meeting summary. They don’t really connect to actual policy thinking. If you don’t remove them systematically, models spend all their attention on these repetitive patterns, and the real language that shows hawkish vs dovish differences gets drowned out. That was the moment it really hit me: **cleaning strength has to match what the document is actually for**. Minutes are internal records — procedural stuff takes up most of the space. Without aggressive, targeted removal, the important signals simply can’t get through.

**Solution and Validation** We built a multi-category, wide-coverage boilerplate dictionary (more than 10 major categories, hundreds of carefully tuned regular expressions) to systematically wipe out those procedural parts. The cleaning is quite aggressive — usually keeping only 40–50% of the original length. After this, the focus of analysis shifted sharply to participants’ views, economic divergences, risk assessments, and so on. Themes became much clearer, and downstream classification performance improved noticeably. This step confirmed what I had started to suspect: **targeted heavy removal works far better than just trying to “clean deeply” in a generic way**.

Code Example: Minutes boilerplate dictionary (partial)
```
 MINUTES_BOILERPLATE = {
    "titles": [
        r"Review of Monetary Policy Strategy, Tools, and Communications",
        r"Staff Review of the Economic Situation",
        r"Committee Policy Action",
        # ... more titles and sections
    ],
    "staff_review_openers": [
        r"The information reviewed at the .*? meeting indicated that",
        r"Staff judged that",
        # ...
    ],
    "manager_reports": [
        r"manager reported .*? developments",
        r"outright purchases",
        r"reinvesting proceeds",
        # ... Desk operations related
    ],
    "liquidity_facilities": [
        r"discount window",
        r"term auction facility taf",
        # ...
    ],
    # Total 10+ categories, including voting_patterns, policy_directives, etc.
 }
```

Cleaning function call (broad removal)

```
 for patterns in MINUTES_BOILERPLATE.values():
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.I | re.DOTALL)
```

##Statement: What Do We Lose When We Over-Clean?##

**Problem** Statements are very short (only a few hundred words) and made up almost entirely of carefully worded policy language. Their real value is in the subtle choices of words — hawkish/dovish turns, changes in forward guidance tone, dual-mandate phrasing, and so on. When I first tried applying the same heavy cleaning style as for Minutes, important signals got removed by accident far too often, and the model could no longer reliably pick up the Fed’s “official final position.”

**Reflection** I originally tried to use one single cleaning approach for both, but I quickly saw that the information density and value distribution are completely different between the two types. Minutes are internal records — you can cut procedural content aggressively. Statements are public statements — every sentence has been weighed carefully. **Chasing “maximum cleanliness” can actually destroy the core value**. That made me ask myself: where should the cleaning stop? We have the technical ability to delete much more, but if we remove the most important “how it’s said,” we’re defeating the whole purpose. The real difficulty isn’t “how much can we delete?” — it’s “what absolutely has to stay?”

**Solution and Validation** We created a lean, highly targeted boilerplate library (only 8 small categories), focusing on fixed opening sentences, mandate restatements, risk assessments, forward guidance commitments, etc. Removal is kept conservative — typically keeping 30–60% or more of the original text. As a result, the core tonal shifts and policy intent were fully preserved, and stance classification accuracy came back strong. This experience really drove home the point: **in high-density documents, restraint is more important than aggressive cleaning.**

Code Example: Statement boilerplate dictionary (partial)
```
 STATEMENT_BOILERPLATE = {
    "opening_summary": [
        r"information received since the federal open market committee met in",
        r"information received since the committee met",
    ],
    "mandate_patterns": [
        r"the committee seeks to foster maximum employment and price stability",
        r"the committee is strongly committed to supporting maximum employment and returning inflation to its 2 percent objective",
    ],
    "risk_assessment": [
        r"uncertainty about the economic outlook remains elevated",
        # ...
    ],
    # Only 8 categories, covering opening, mandate, risk, forward guidance, etc.
 }
```

Cleaning function call (precise, limited removal)

```
 for patterns in STATEMENT_BOILERPLATE.values():
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.I | re.DOTALL)
```

##Shared General Modules: Principles Learned from Mistakes##

1. **Policy Phrase MergingProblem**: Multi-word policy terms got split apart, losing semantic meaning. **Reflection**: At first I didn’t realize how important keeping phrases intact really was. **Solution**: We merged them into single tokens (e.g., "federal_funds_rate"). It took almost no effort but dramatically improved feature quality.

2. **Verb NormalizationProblem**: Stemming created nonsense words and wiped out tense nuance, destroying signals about policy expectations. **Reflection**: Automated tools can easily backfire in central bank text; subtle language choices are signals in themselves. **Solution**: Switched to manual mapping (e.g., "promoting"/"fostering" → "support"). It takes more time, but it’s safer and helped me understand Fed expression patterns much better.

3. **Aggressive Number & Time FilteringProblem**: Numbers completely dominated the features and masked qualitative signals. **Reflection**: Numbers look like “obvious” information, but in Fed communication the real insight is in how things are described, not the exact number. The ECB paper Monetary Communication Rules reinforced this for me. **Solution**: Completely removed all numerics and dates. The model was forced to learn implicit signals, and classification performance improved a lot.

4. **Expanded Stop WordsProblem**: Generic stop words weren’t sensitive to financial text. **Reflection**: Off-the-shelf NLP tools often fall short in specialized domains. **Solution**: Added high-frequency/low-information Fed terms (committee, vote, etc.) while keeping meaningful connectors like "to".

##Closing Thoughts & Outlook##

This whole process taught me the most important lesson: **text cleaning is really about understanding the document, not just applying technical tricks**. Moving from trying to use one uniform method to accepting different strategies for different documents, from depending on automation to adding manual judgment and restraint — every change came from looking back at the data and questioning what I had assumed. The cleaner text we ended up with now gives a much stronger foundation for everything that comes next.
Looking ahead, we might try using LLMs to help generate boilerplate patterns and reduce manual work. But the core principle won’t change: **understand the document first, then figure out how to clean it**.



## Feature Extraction

After data preprocessing, we focused on feature extraction from FOMC statements and meeting minutes. Quantitative text-based features were constructed using dictionary-based measures, TF-IDF representations, and embedding to capture economic stance, as well as lexical and semantic information, for downstream prediction tasks.

##Dictionary-Based Approach##
For this part, we constructed several dictionaries to capture economically meaningful signals in FOMC communications, including hawkish–dovish, uncertainty, inflation focus and labor focus. As there has been no standardized framework for constructing such dictionaries, we relied on prior literature, official documents, and economic intuition to guide the dictionary design. Among the dictionaries, we deemed the hawkish–dovish dictionaries as the most central, as they directly reflect the monetary policy stance relevant for interest rate decisions. We first constructed the hawkish–dovish dictionaries based on single words, but then realized that complex multi-word expressions can be of great importance to deciphering hawkish or dovish signals, as texts might contain confusing expressions such as ‘rise of unemployment rate’, in which ‘rise’ is believed by some literature to display a hawkish stance, while the whole expression often leans more to the dovish side. Therefore, in the following program we chose to add a second layer to our dictionary structure: phrase-level matching is applied first, followed by word-level counting, to better capture nuanced policy signals.


```
    def mask_phrases(text, phrases):
        masked = text
        for p in phrases:
            pattern = re.escape(p)
            masked = re.sub(
                rf"\b{pattern}\b",
                " ",
                masked,
                flags=re.IGNORECASE
            )
        return masked
    def stance_score(text, dovish_phrases, hawkish_phrases, dovish_words, hawkish_words):
        # 1. phrase-level
        dove_p = sum(text.count(p) for p in dovish_phrases)
        hawk_p = sum(text.count(p) for p in hawkish_phrases)
        # 2. mask phrases
        masked_text = mask_phrases(text, dovish_phrases + hawkish_phrases)
        masked_tokens = masked_text.split()
        # 3. word-level
        dove_w = sum(
            1 for t in masked_tokens
            if any(t.startswith(s) for s in dovish_words)
        )
        hawk_w = sum(
            1 for t in masked_tokens
            if any(t.startswith(s) for s in hawkish_words)
        )
        
        hawk_total = hawk_p + hawk_w
        dove_total = dove_p + dove_w
        
        return pd.Series({
            "hawk_score": hawk_total / (hawk_total + dove_total) if (hawk_total + dove_total) > 0 else 0,
            "dove_score": dove_total / (hawk_total + dove_total) if (hawk_total + dove_total) > 0 else 0,
            "net_stance": (hawk_total - dove_total) / (hawk_total + dove_total) if (hawk_total + dove_total) > 0 else 0  
    })
```

To our surprise, some features from the two documents showed specific diverge in their levels and trends. For example, statements have displayed much larger fluctuations and more extreme stances than minutes over the past fifteen years. We believe this may be due to the fact that statements are intentionally brief and designed to send clear signals to markets, which may encourage stronger language. In contrast, minutes summarize a wide range of internal discussions, leading to more measured and stable textual signals. This difference gave us a better understanding of how the purpose of a document shapes the language used in monetary policy communication, and how it affects our interpretation.

##TF-IDF##
In addition to dictionary-based features, we used TF-IDF to capture terms and phrases that are informative but cannot be explicitly encoded in predefined dictionaries. By doing so, we aim to further highlight important language patterns in FOMC communications in a more flexible, data-driven way. Due to the small sample size, we paid high awareness to the parameters in order to control dimensionality and reduce noise, with the parameters showcasing our trade-off between capturing meaningful variation in language and maintaining model stability.


```
    def build_tfidf_features(
        texts,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        max_features=3000
    ):
        
        vectorizer = TfidfVectorizer(
            lowercase=False,      # text already normalized
            stop_words=None,      # stopwords removed in preprocessing
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=vectorizer.get_feature_names_out()
        )
        
    return tfidf_matrix, tfidf_df
```

The results revealed both key policy concepts and actionable measures, providing a strong feature foundation for predicting the next FOMC policy decision. They also strongly validated the effectiveness of our data preprocessing and dictionary-based approach. We were able to acquire insights from these results, so that we could improve our preprocessing methods and dictionary construction.

##Text similarity: TF–IDF and embedding##

To further understand how FOMC communication evolves over time, we constructed text similarity measures using both TF-IDF and embedding. TF-IDF results were used to capture lexical overlap, reflecting how much wording and phrasing is reused across documents. In parallel, we also applied a pre-trained MiniLM variant to compute embedding-based similarity, which captures semantic similarity even when exact words differ. The motivation behind using both measures was to distinguish changes in surface language from deeper shifts in meaning.


```
    def similarity_tfidf(texts, tfidf_matrix):
    similarity_tfidf = [np.nan] 
    for i in range(1, tfidf_matrix.shape[0]):
    	sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i-1])[0, 0]
    	similarity_tfidf.append(sim)
    return similarity_tfidf

    def similarity_embedding(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = texts.tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    similarity_embedding = [np.nan]
    for i in range(1, len(embeddings)):
        sim = np.dot(embeddings[i], embeddings[i-1])
        similarity_embedding.append(sim)
    return similarity_embedding
```

The results gave us a deeper understanding of how the communications had changed over time. We found that FOMC statements generally exhibited higher similarity over time than minutes under both measures, and that lexical and semantic similarities for statements moved comparatively more closely together. Both aligned with our previous understanding of the two document types: statements are carefully crafted policy signals with relatively stable structure, whereas minutes summarize diverse internal discussions and therefore display greater variation in their meanings. Combined with the dictionary-based results showing more pronounced stance signals in statements, we believe that even relatively small changes in the wording or semantics of statements may convey meaningful information about federal reserve’s decision intentions.




## Model Training

In this section, we will train four models on the statement and minutes datasets using both word set method and TD-IDF method. These four models are:logistic,random_forest,gradient_boosting,svm


##Method 1: Dictionary-based Approach##

Core Concept
The dictionary method uses pre-defined hawkish and dovish dictionaries to quantify policy stance by calculating the frequency of these words in the text.


We use four different machine learning models, After debugging, the parameter settings are as follows::

```
 def train(self, X_train, y_train, model_type='random_forest'):
        """Train the model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if model_type == 'logistic':
            model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', probability=True)
        
        model.fit(X_train_scaled, y_train)
 return model
```

##Method 2: TF-IDF-based Approach##

Core Concept
The TF-IDF (Term Frequency-Inverse Document Frequency) method constructs high-dimensional feature vectors by calculating the importance of words across the entire document collection, capturing deep semantic information of the text.


Feature Selection
Since TF-IDF features have high dimensionality (typically thousands of dimensions), we use SelectKBest for feature selection, retaining the most important 50 features:

```
 def select_features(self, X, y):
        """Feature selection"""
        X_selected = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.get_support()
        return X_selected
    Experimental Design and Evaluation
    Data Splitting
    Considering time series characteristics, we split the training and test sets in chronological order:


 def prepare_train_test_split(df, features, test_size=0.2):
        """Prepare train-test split (considering time series characteristics)"""
        split_idx = int(len(df) * (1 - test_size))
        
        X = features
        y = df['target'].values
        
        # Time series split
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
    return X_train, X_test, y_train, y_test
```

##Data partitioning##
We use multiple evaluation metrics to comprehensively assess model performance:

```
 def evaluate_model(y_true, y_pred, y_proba=None, model_name=""):
    """Evaluate model performance"""
        results = {}
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # ROC-AUC (if multi-class probabilities available)
        if y_proba is not None:
            results['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        
 return results
```

##Experimental Results Analysis##

##Statement Analysis Results##
For FOMC Statements, the TF-IDF method significantly outperforms the dictionary method across all models:

```
    - Random Forest:
    Dictionary method accuracy: 0.269,
    TF-IDF method accuracy: 0.769,
    Better method: TF-IDF method (difference: 0.500)

    - Logistic Regression:
    Dictionary method accuracy: 0.462,
    TF-IDF method accuracy: 0.962,
    Better method: TF-IDF method (difference: 0.500)

    - Gradient Boosting:
    Dictionary method accuracy: 0.423,
    TF-IDF method accuracy: 0.769,
    Better method: TF-IDF method (difference: 0.346)

    - SVM:
    Dictionary method accuracy: 0.615,
    TF-IDF method accuracy: 0.923,
    Better method: TF-IDF method (difference: 0.308)
```

##Minutes Analysis Results##
For FOMC Minutes, the results are more complex, with different models performing differently:

```
    - Random Forest:
    Dictionary method accuracy: 0.615,
    TF-IDF method accuracy: 0.577,
    Better method: Dictionary method (difference: 0.038)

    - Logistic Regression:
    Dictionary method accuracy: 0.192,
    TF-IDF method accuracy: 0.654,
    Better method: TF-IDF method (difference: 0.462)

    - Gradient Boosting:
    Dictionary method accuracy: 0.538,
    TF-IDF method accuracy: 0.692,
    Better method: TF-IDF method (difference: 0.154)

    - SVM:
    Dictionary method accuracy: 0.462,
    TF-IDF method accuracy: 0.231,
    Better method: Dictionary method (difference: 0.231)
```

###Key Findings###
1. Statement vs Minutes: TF-IDF method performs significantly better on Statements, but both methods have their advantages on Minutes

2. Model Selection: Logistic regression achieves an impressive 96.2% accuracy on Statements, but shows unstable performance on Minutes

3. Method Stability: Random forest shows relatively stable performance in both methods

##Conclusion##

This article comprehensively compares dictionary-based and TF-IDF methods for FOMC monetary policy forecasting. Experimental results show:

1. The TF-IDF method generally outperforms the dictionary method, particularly for Statement prediction

2.Different document types require different modeling strategies. Model selection significantly impacts prediction performance

These findings provide important empirical evidence for the fintech field and point directions for future research. The code implementation is open-sourced, and we welcome researchers to further improve and extend it.



## Results Summarize

Evaluating Models on Fed Meeting Records

Before starting the evaluation, I honestly didn’t hold much confidence in the outcomes. During the initial topic selection, we hesitated over alternatives—mainly because the available sample size for this project was relatively small. For training large language models or similar AI systems, limited data often risks introducing bias. While organizing the dataset, we also noticed that most of the existing meeting records corresponded to a “hold” trend in interest rates. After all, governments don’t adjust rates with high frequency.

Among the 16 models, surprisingly, the four selected models—fixed under specific document and method categories—performed better than expected. In particular, the logistic regression model based on Statement documents and the TF-IDF method achieved **the best performance with 0.962 accuracy**. We believe this is largely due to the nature of statements themselves: short text length and clear keywords make them highly compatible with logistic regression. However, we cannot rule out the possibility of overfitting given the small sample size.

We initially planned to compare our results with CME forecasts, but the difference in sample scope made such a comparison less meaningful. We also wrote code for robustness checks, though without additional samples to test against, we only retained the framework without executing it.

Overall, we were pleased to see that four of our selected models reached promising levels of accuracy and performance. This suggests that **Fed meeting transcripts can indeed be leveraged to predict future interest rate movements**. The analysis of document characteristics, which provided valuable insights for choosing different models in future prediction tasks.

```
 def evaluate_model(y_true, y_pred, y_proba=None, model_name=""):
    results = {}
    
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(unique_classes)
    
    class_mapping = {-1: 'cut', 0: 'hold', 1: 'hike'}
    available_classes = [class_mapping[c] for c in sorted(unique_classes) if c in class_mapping]
    
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    try:
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    except:
        class_report = {}
    
    # ROC-AUC
    if y_proba is not None and num_classes > 1:
        try:
            if num_classes == 2:
                results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                results['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                )
        except:
            results['roc_auc'] = np.nan
            results['roc_auc_ovr'] = np.nan
    
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"model evaluation: {model_name}")
    print(f"{'='*60}")
    print(f"class numbers: {num_classes}")
    print(f"actual class: {sorted(unique_classes)}")
    print(f"accuracy: {results['accuracy']:.4f}")
    print(f"precision_macro: {results['precision_macro']:.4f}")
    print(f"recall_macro: {results['recall_macro']:.4f}")
    print(f"f1_macro: {results['f1_macro']:.4f}")
    
    if 'roc_auc' in results and not np.isnan(results['roc_auc']):
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
    elif 'roc_auc_ovr' in results and not np.isnan(results['roc_auc_ovr']):
        print(f"ROC-AUC (OVR): {results['roc_auc_ovr']:.4f}")
    
    print("\n Classification Report:")
    try:
        #the available category names
        print(classification_report(
            y_true, y_pred, 
            target_names=available_classes,
            zero_division=0
        ))
    except:
        #If it still fails, use the numeric labels.
        print(classification_report(y_true, y_pred, zero_division=0))
    
    return results, cm

def plot_confusion_matrix(cm, model_name, class_names=None):
    """Plot confusion matrix"""
    if class_names is None:
        n_classes = cm.shape[0]
        if n_classes == 3:
            class_names = ['Cut', 'Hold', 'Hike']
        elif n_classes == 2:
            if set([-1, 0]).issubset(set(range(-1, 2))):
                class_names = ['Cut', 'Hold']
            elif set([0, 1]).issubset(set(range(-1, 2))):
                class_names = ['Hold', 'Hike']
            elif set([-1, 1]).issubset(set(range(-1, 2))):
                class_names = ['Cut', 'Hike']
            else:
                class_names = [f'Class{i}' for i in range(n_classes)]
        else:
            class_names = [f'Class{i}' for i in range(n_classes)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()



def plot_feature_importance(importance_df, model_name, top_n=20):
    """Plot feature importance"""
    if importance_df is not None:
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'], color='gray')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


    #Multi-source robustness test (lack of data)
    #print(robustness_df)

    #Define the robustness test function
    #from scipy.stats import ttest_ind, ks_2samp
    #import pandas as pd

    #def generate_robustness_report(df_all, metrics=["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)", "ROC-AUC (OVR)"]):
    #    results = []
    #    for metric in metrics:
    #        s_vals = df_all[df_all["Document"]=="Statement"][metric].dropna()
    #        m_vals = df_all[df_all["Document"]=="Minutes"][metric].dropna()
            
    #        if len(s_vals) > 1 and len(m_vals) > 1:
    #            t_stat, t_p = ttest_ind(s_vals, m_vals, equal_var=False)
    #            ks_stat, ks_p = ks_2samp(s_vals, m_vals)
    #        else:
    #            t_stat, t_p, ks_stat, ks_p = [None]*4
            
    #        results.append({
    #            "Metric": metric,
    #            "Statement Mean": s_vals.mean() if len(s_vals)>0 else None,
    #            "Minutes Mean": m_vals.mean() if len(m_vals)>0 else None,
    #            "Welch t-stat": t_stat,
    #            "t-test p-value": t_p,
    #            "KS stat": ks_stat,
    #            "KS p-value": ks_p
    #        })
    #    return pd.DataFrame(results)

    # Call the function DataFrame
    #robustness_df = generate_robustness_report(df_all)
    #print(robustness_df)

    # Write the robustness test results into Excel
    #robustness_df.to_excel("comparison result.xlsx", sheet_name="Robustness Test", index=False)
```



This study demonstrates that semantic and lexical approaches not only enhance model interpretability but also more robustly capture key expressions of policy shifts when processing highly structured, policy-driven texts. While TF-IDF excels in general text classification, in contexts where language directly reflects policy (e.g., central bank language), subtle contextual and lexical variations often outweigh word frequency.
This study not only provides an effective framework for analyzing central bank communications, but also offers novel textual analysis approaches for policy forecasting, market sentiment analysis, and even macroeconomic modeling. In the future, a hybrid method combining deep learning and policy lexicons may further optimize the balance between prediction accuracy and semantic depth.
Thank you for reading. We welcome readers interested in central bank text analysis and policy forecasting to follow our ongoing research.