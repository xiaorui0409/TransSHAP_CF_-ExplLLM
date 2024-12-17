# TransSHAP_CF_-ExplLLM
**1.Background:**
Large Language Models (LLMs) are increasingly applied across fields like healthcare and finance. However, challenges such as hallucinations and limited reasoning capabilities remain critical. To address these issues, we aim to combine **TransSHAP** and **Counterfactual Testing** to explain **LLM** decision-making and identify the key information influencing predictions.

**2.TransSHAP:**
**TransSHAP** is an adaptation of the classical SHAP (SHapley Additive exPlanations) method, specifically designed for transformer-based models like BERT. Unlike traditional SHAP implementations, TransSHAP accounts for the unique attention mechanism in transformer architectures and maintains the semantic relationships between words.  ### References
[BERT meets Shapley: Extending SHAP Explanations to Transformer-based Classifiers](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://aclanthology.org/2021.hackashop-1.3.pdf)
The figure illustrates the components necessary for TransSHAP to produce explanations for predictions made by the BERT model.
Implementation Pipeline for TransSHAP
1.	**Word-Level Tokenization and Perturbation:**
o	Use a word-level tokenizer (e.g., Twitter Tokenizer) to split the raw text into individual words.
o	Apply perturbation operations at the word level, such as masking or word replacement, to ensure interpretability for human understanding.
2.	**Subword Tokenization for Model Input**:
o	Tokenize the perturbed sentences using the BERT tokenizer, which splits words into subword components to meet the input requirements of the BERT model.
3.	**Model Prediction and SHAP Value Computation**:
o	Pass the subword tokenized sentences into the BERT model to generate predictions.
o	Use Kernel SHAP to compute the contribution of each subword token to the prediction outcome.
4.	**Aggregation of SHAP Values to Word Level**:
o	Aggregate the computed SHAP values of the subword components back to their original words.
o	This step ensures that the explanations are interpretable at the word level while retaining the model’s subword-level precision.
5.	**Visualization of Explanations**:
o	Map the aggregated SHAP values onto the original sentence, preserving the sequential order of words.
5.	Visualization of Explanations:
o	Map the aggregated SHAP values onto the original sentence, preserving the sequential order of words.

**3.Counterfactual testing**
**Counterfactual Testing for NLP Models**
Traditional methods for generating counterfactuals, such as replacing words with synonyms or masking specific words, often have notable limitations. These methods produce counterfactuals that lack diversity and frequently disrupt the semantics of the original sentence, leading to unrealistic or irrelevant outputs
To address these issues, the author leverage Large Language Models (LLMs) for counterfactual generation in a zero-shot manner. The method involves two steps:
1.	Using carefully designed **prompts** to instruct the LLM to **identify key keywords** that influence the model’s prediction.
2.	**Minimally editing** these identified keywords to generate **new counterfactuals** that successfully flip the label while preserving the sentence's meaning.
[Zero-shot LLM-guided Counterfactual Generation: A Case Study on NLP Model Evaluation](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2405.04793)  


**4.Integration of the TransSHAP and Counterfactual testing to enhance the explainability of LLM**

**Hypothesis 1: Target Word-focused Perturbation in SHAP**
**limitation**:The classical SHAP computation uses random sampling for perturbations, which can lead to two significant limitations:
**limitation1: Ignoring Important Words**: Random sampling might exclude key words that contribute the most to a model’s prediction.

Example: In the sentence “I like dog, because it was my friend”, random sampling might select words like "it" or "my", while important words such as "dog" and "friend" are ignored. As a result, SHAP may fail to capture their true importance.

**limitation2:Contribution Shrinkage**: Even when important words are included in randomly selected subsets, the presence of irrelevant or unimportant words reduces their contribution. This shrinks the SHAP value of the key words and increases computational demand unnecessarily.
Example: If the subset {"it", "my"} is perturbed along with "dog", the overall contribution might appear small because "it" and "my" contribute nothing. This masks the true impact of "dog".

**Proposed Solution**:
To address these limitations, I propose performing perturbations **exclusively on target words** identified as important features using the FIZLE-guided method. This ensures that:

SHAP values for important words reflect their true contribution.
Computational efficiency improves by avoiding irrelevant perturbations.

**Hypothesis 2: Intersection/Union of Keywords for Counterfactual Generation**
The goal is to generate high-quality counterfactuals that meet the following criteria:
**Minimal Edits**: Small changes to the original text.
**Label Flip**: The modified text successfully changes the prediction label.
**Semantic Similarity**: The generated counterfactual remains close in meaning to the original sentence.

**Proposed Solution**:
I propose using the **intersection or union** of keywords extracted via SHAP and FIZLE-guided methods to generate counterfactuals. The combined keywords focus on critical words that influence predictions the most.

If the generated counterfactuals using this approach result in fewer edits and better semantic similarity compared to those generated solely by the FIZLE method, it demonstrates that the extracted keywords have significant impact on the model’s prediction.

Both approaches aim to enhance the interpretability and evaluation of model predictions by:
Improving the accuracy and stability of SHAP explanations.
Generating more effective and high-quality counterfactual examples.
