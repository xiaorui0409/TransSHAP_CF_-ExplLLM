# **TransSHAP_CF - ExpLLM**

## **1. Background**  
Large Language Models (LLMs) are undeniably a prominent area of interest and have been widely applied across various domains, including healthcare and finance. However, challenges like hallucinations and limited reasoning abilities persist. To tackle these challenges, we propose **integrating TransSHAP and Counterfactual Testing** to enhance the explainability of LLM decisions and identify the critical information driving their predictions.

---

## **2. TransSHAP**  
**TransSHAP** is an adaptation of the classical **SHAP (SHapley Additive exPlanations)** method, specifically designed for transformer-based models like BERT. Unlike traditional SHAP implementations, TransSHAP accounts for the unique attention mechanism in transformer architectures and maintains the semantic relationships between words.  

This approach was taken from the paper:  
[BERT meets Shapley: Extending SHAP Explanations to Transformer-based Classifiers](https://aclanthology.org/2021.hackashop-1.3.pdf) by Kokalj et al., 2021.

---

## **3. Counterfactual Testing for NLP Models**  
Traditional counterfactual generation methods, like synonym replacement or word masking, often produce limited, unrealistic outputs that disrupt the original sentence's semantics. To address these issues, the authors leverage **Large Language Models (LLMs)** for counterfactual generation in a **zero-shot manner**.  

The **FIZLE-guided method** involves two steps (Bhattacharjee et al., 2024):  
1. **Using carefully designed prompts** to instruct the LLM to **identify key keywords** that influence the model’s prediction.  
2. **Minimally editing** these identified keywords to generate **new counterfactuals** that successfully flip the label while preserving the sentence's meaning.  

[Read the full paper here](https://arxiv.org/abs/2405.04793).

---

## **4. Integrating TransSHAP and Counterfactual Testing to Improve LLM Explainability**  

### **Proposed Solution 1: Target Word-focused Perturbation in SHAP**  
We propose performing perturbations **exclusively on target words** identified as important features using the **FIZLE-guided method**. This ensures that:  
- SHAP values for important words reflect their **true contribution**.  
- Computational efficiency improves by avoiding irrelevant perturbations.  

#### **Explanation**  
The classical SHAP computation uses random sampling for perturbations, which can lead to two significant limitations:  

1. **Ignoring Important Words**  
   - **Issue**: Random sampling might exclude key words that contribute the most to a model’s prediction.  
   - **Example**: In the sentence *“I like dog, he is my best friend”*, random sampling might select words like `"is"` or `"my"`, while important words such as `"dog"` and `"friend"` are ignored. As a result, SHAP may fail to capture their true importance.  

2. **Contribution Shrinkage**  
   - **Issue**: Even when important words are included in randomly selected subsets, the presence of irrelevant or unimportant words reduces their contribution.  
   - **Example**: If the subset `{"is", "my"}` is perturbed along with `"dog"`, the overall contribution might appear small because `"is"` and `"my"` contribute nothing. This **masks the true impact** of `"dog"` and `"friend"`.  

---

### **Proposed Solution 2: Intersection/Union of Keywords for Counterfactual Generation**  
We propose using the **intersection or union** of keywords extracted via **SHAP** and **FIZLE-guided methods** to generate counterfactuals. The combined keywords focus on critical words that influence predictions the most.

If the generated counterfactuals using this approach result in **fewer edits** and **better semantic similarity** compared to those generated solely by the FIZLE method, it demonstrates that the extracted keywords have a significant impact on the model’s prediction.

#### **Goals**  
The goal is to generate **high-quality counterfactuals** that meet the following criteria:  
- **Minimal Edits**: Small changes to the original text.  
- **Label Flip**: The modified text successfully changes the prediction label.  
- **Semantic Similarity**: The generated counterfactual remains close in meaning to the original sentence.  

---

## **5. Conclusion**  
Both approaches aim to enhance the interpretability and evaluation of model predictions by:  
1. Improving the **accuracy and stability** of SHAP explanations.  
2. Generating more **effective and high-quality counterfactual examples**.

**Semantic Similarity**: The generated counterfactual remains close in meaning to the original sentence.


Both approaches aim to enhance the interpretability and evaluation of model predictions by:
Improving the accuracy and stability of SHAP explanations.
Generating more effective and high-quality counterfactual examples.
