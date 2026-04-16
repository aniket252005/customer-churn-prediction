# 📊 Business Insights — Customer Churn Analysis
*IBM Telco Dataset · 7,043 customers · XGBoost AUC 0.8469 · Recall 0.8021*

---

## 🧠 Executive Summary

26.54% of Telco customers churn annually. This analysis identifies **who churns, why they churn, and what it costs** — using machine learning (XGBoost), SHAP explainability, and segment-level business analysis. The output is a ranked list of high-risk customers with actionable retention strategies for each risk tier.

**Model selected for production:** XGBoost — AUC 0.8469, Recall 0.8021
The model correctly identifies 80 out of every 100 actual churners before they leave.

---

## 🔍 Business Insight 1 — Contract Type is the Strongest Churn Driver

| Contract Type | Churn Rate | vs Two-Year |
|---|---|---|
| Month-to-month | 42.71% | **15× higher** |
| One year | 11.27% | 4× higher |
| Two year | 2.83% | baseline |

**Finding:** A month-to-month customer is 15 times more likely to churn than a two-year contract customer. This single feature has the highest SHAP impact on the model — `Contract_Two year` pushes churn probability down by an average of **−1.79 units** per customer (largest magnitude in the waterfall plot).

**Revenue impact:** 100% of the high-risk revenue ($23.22K) is concentrated in month-to-month customers.

**Action:** Offer a discounted rate or one-time credit to migrate month-to-month customers to annual plans. Even converting 10% of high-risk month-to-month customers would protect approximately $2,300 in revenue.

---

## 🔍 Business Insight 2 — The First 12 Months is the Highest-Risk Window

| Tenure Group | Churn Rate |
|---|---|
| 0–12 months | **48%** |
| 12–24 months | 29% |
| 24–48 months | 20% |
| 48–72 months | 10% |

**Finding:** Nearly half of all new customers (0–12 months) churn before their first anniversary. Churn drops by 5× between the earliest and latest tenure groups. SHAP confirms tenure as the second-most impactful feature — low tenure pushes churn probability up by +1.25 units on average.

**Action:** Build a 90-day onboarding program targeting customers in month 1–3. Automated check-ins, proactive tech support outreach, and service bundle offers during this critical window can significantly reduce early churn.

---

## 🔍 Business Insight 3 — Fiber Optic Internet is a Hidden Churn Risk

| Internet Service | Churn Rate | vs No Internet |
|---|---|---|
| Fiber optic | 41.89% | **5.7× higher** |
| DSL | 18.96% | 2.6× higher |
| No internet | 7.40% | baseline |

**Finding:** Fiber optic customers churn at 2.2× the rate of DSL users despite paying premium prices. SHAP beeswarm confirms `InternetService_Fiber optic` as the 3rd most impactful feature — high fiber optic usage pushes churn probability upward (+0.19 in waterfall example). This is a pricing/value mismatch signal: customers paying more expect more, and when expectations aren't met, they leave.

**Revenue impact:** Fiber optic customers account for $23.2K of the total $23.22K revenue at risk — virtually all high-risk revenue sits in this segment.

**Action:** Prioritize service quality improvement and proactive support for fiber optic customers. Consider a "Fiber Optic Loyalty" bundle pairing high-speed internet with tech support and online security at a fixed price.

---

## 🔍 Business Insight 4 — Electronic Check Payment is a Churn Predictor

| Payment Method | Churn Rate |
|---|---|
| Electronic check | **45.29%** |
| Mailed check | 19.11% |
| Bank transfer (auto) | 16.71% |
| Credit card (auto) | 15.24%  |

**Finding:** Electronic check users churn at nearly 3× the rate of customers on automatic payment methods. SHAP confirms `PaymentMethod_Electronic check` as the 6th-ranked feature — False (not using electronic check) reduces churn risk by −0.12 per customer. Electronic check is a manual, friction-heavy payment method; customers using it are likely less committed and less engaged.

**Action:** Run an auto-pay migration campaign. Offer a $5–10/month discount for switching to credit card or bank transfer auto-pay. Lower payment friction → higher retention.

---

## 🔍 Business Insight 5 — Lack of Support Services Drives Exits

| Tech Support | Churn Rate |
|---|---|
| No tech support | **41.64%** |
| Has tech support | 15.17% |
| No internet service | 7.40% |

**Finding:** Customers without tech support churn at 2.7× the rate of those who have it. Similarly, customers without online security churn at 41%+ vs 14% with it. SHAP places `OnlineSecurity_Yes` (−0.30) and `TechSupport_Yes` (−0.13) as protective features — both reduce churn probability when present.

**Action:** Bundle tech support and online security as free add-ons for the first 6 months for new customers. The acquisition cost of these services is far lower than the LTV of a churned customer.

---

## 💰 Financial Impact Summary

| Metric | Value |
|---|---|
| Total customers analyzed | 7,043 |
| Overall churn rate | 26.54% |
| Customers churning (est.) | ~1,869 |
| Revenue at risk (high-risk segment) | $23,220 |
| Model Recall | 80.21% — catches 8 in 10 actual churners |
| Model AUC | 0.8469 — strong discrimination ability |

**Conservative retention scenario:** If targeted outreach converts 25% of model-identified high-risk customers from churning, and average monthly revenue per churned customer is $80, that's approximately **$5,800 in protected annual revenue** from the flagged segment alone.

---

## 🎯 Retention Strategy by Risk Tier

### 🔴 High Risk (Churn Probability ≥ 50%)
- Phone outreach within 7 days
- Offer contract upgrade with 15–20% discount
- Assign dedicated account manager if MonthlyCharges > $80

### 🟡 Medium Risk (Churn Probability 30–50%)
- Targeted email campaign within 14 days
- Bundle offer: tech support + online security at no additional charge
- Auto-pay migration incentive ($5/month discount)

### 🟢 Low Risk (Churn Probability < 30%)
- Standard engagement — quarterly satisfaction survey
- Loyalty reward at 12-month and 24-month anniversaries
- Upsell opportunity — these customers are stable and receptive

---

## 🔬 SHAP Model Explainability Summary

Top features ranked by absolute SHAP impact (global, across all customers):

| Rank | Feature | Direction | Business Meaning |
|---|---|---|---|
| 1 | `Contract_Two year` | ↓ reduces churn | Long contracts = loyalty |
| 2 | `tenure` | ↓ reduces churn (high value) | Longer customers are safer |
| 3 | `InternetService_Fiber optic` | ↑ increases churn | Premium service, unmet expectations |
| 4 | `Contract_One year` | ↓ reduces churn | Annual contracts still protective |
| 5 | `TotalCharges` | ↓ reduces churn (high value) | High spend = long tenure proxy |
| 6 | `PaymentMethod_Electronic check` | ↑ increases churn | Manual payment = low commitment |
| 7 | `MonthlyCharges` | ↑ increases churn (high value) | High cost without perceived value |
| 8 | `OnlineSecurity_Yes` | ↓ reduces churn | Service stickiness |
| 9 | `TechSupport_Yes` | ↓ reduces churn | Support = retention |

**Key SHAP takeaway:** The model's logic is entirely business-rational. Every top feature maps directly to a known churn driver. This means the model can be trusted for production deployment and its recommendations can be explained to non-technical stakeholders without qualification.

---

## 🚀 Conclusion

This project demonstrates that churn is **predictable and preventable**. The top risk factors — contract type, early tenure, fiber optic without support services, and manual payment methods — are all addressable through targeted interventions. A model with 80.21% Recall means 8 out of 10 customers who would have churned can now be identified and approached before they leave.

The combination of ML prediction + SHAP explainability + Power BI dashboard creates a complete retention intelligence system that a business team can act on directly, without needing to understand the underlying model.

---

*Generated from XGBoost model · IBM Telco Dataset · April 2026*