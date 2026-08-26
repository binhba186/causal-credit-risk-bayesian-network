#  Causal-BN Credit Risk — Bayesian Network cho Đánh giá Rủi ro Tín dụng & Phân tích Can thiệp

> **"Từ tương quan đến nguyên nhân: mô hình hóa rủi ro vỡ nợ bằng cấu trúc nhân quả tường minh, không chỉ dự đoán mà còn giải thích *tại sao* và *nếu-thì*."**

## Executive Summary

### Bài toán
Các mô hình chấm điểm tín dụng (credit scoring) truyền thống — Logistic Regression, Random Forest, XGBoost — tối ưu hóa khả năng **dự đoán** `P(default | X)` dựa trên tương quan thống kê giữa đặc trưng và biến mục tiêu. Điều này tạo ra hai rủi ro nghiêm trọng trong thực tế vận hành:

1. **Spurious Correlation (Tương quan giả)**: Một mô hình ML thuần túy có thể học được rằng `Ethnicity` hay `Gender` có tương quan với `default`, không phải vì có quan hệ nhân quả, mà vì các biến này cùng chịu ảnh hưởng bởi một biến ẩn khác (ví dụ: khu vực địa lý, nhóm nghề nghiệp lịch sử). Dùng trực tiếp tương quan này để ra quyết định tín dụng dẫn đến thiên vị thuật toán (algorithmic bias) và vi phạm các quy định công bằng tín dụng (Fair Lending, ECOA).
2. **Không hỗ trợ ra quyết định can thiệp (Actionability Gap)**: Câu hỏi kinh doanh thực sự không phải là "*mô hình dự đoán gì?*" mà là "*nếu tôi tăng hạn mức, giảm lãi suất, hoặc yêu cầu thêm tài sản đảm bảo, xác suất vỡ nợ thay đổi ra sao?*". Một bộ phân loại correlation-based không thể trả lời câu hỏi can thiệp (interventional question) này một cách nhất quán, vì nó không phân biệt được đâu là nguyên nhân, đâu là hệ quả, đâu là confounder.

### Giải pháp: Causal-Inspired Bayesian Network
Dự án xây dựng một **Discrete Bayesian Network (BN)** trong đó cấu trúc đồ thị (DAG) được ràng buộc bởi **tri thức chuyên gia (expert knowledge)** — cấm các cạnh phi lý về mặt nhân quả (ví dụ: `target → Gender`, `target → Age`, `target → CreditHistory`) — nhằm buộc thuật toán học cấu trúc chỉ tìm các quan hệ có hướng **hợp lý về mặt nhân quả** (đặc trưng nhân khẩu học/tài chính là nguyên nhân tiềm năng dẫn đến default, không phải ngược lại). Trên nền tảng đó, hệ thống hỗ trợ:

- **Suy diễn xuôi (Forward/Predictive Inference)**: `P(default | evidence)`.
- **Suy diễn ngược (Backward/Diagnostic Inference)**: `P(feature | default = 1)` — tìm đặc điểm điển hình của nhóm vỡ nợ.
- **Phân tích kịch bản giả định (What-if / Soft Intervention)**: mô phỏng thay đổi từng đặc trưng và đo lường ΔP(default) tương ứng — nền tảng cho các đề xuất chính sách tín dụng.

 **Minh bạch về giới hạn phương pháp**: Bayesian Network trong dự án này thực hiện suy diễn qua **posterior update / soft evidence** bằng thuật toán Variable Elimination, **không phải do-calculus đầy đủ theo nghĩa Pearl** (`P(Y | do(X))`). DAG được ràng buộc bởi tri thức miền để phản ánh hướng nhân quả hợp lý, nhưng chưa xử lý confounding ẩn bằng backdoor/frontdoor adjustment hay ước lượng ATE/CATE dạng đóng (closed-form). Đây là điểm khác biệt quan trọng cần nêu rõ khi trình bày kết quả, và cũng là hướng mở rộng chính trong Roadmap.

---

## Core Methodology

### 1. Structural Bayesian Network như một xấp xỉ Causal DAG

Mô hình được biểu diễn bởi bộ ba $\langle G, \Theta, D \rangle$:

- $G = (V, E)$: đồ thị có hướng phi chu trình (DAG) trên tập biến $V = \{X_1, ..., X_n, Y\}$.
- $\Theta$: tập tham số $\theta_{ijk} = P(X_i = k \mid \text{pa}(X_i) = j)$.
- Ràng buộc chuyên gia (`ExpertKnowledge.forbidden_edges`): loại trừ tiên nghiệm các cạnh $Y \to X_i$ với $X_i$ thuộc nhóm nhân khẩu học (`Gender`, `Age`, `Ethnicity`, `Citizen`, `EducationLevel`) và nhóm lịch sử tài chính (`CreditHistory`, `CreditScore`, `YearsEmployed`, `Debt`, `Income`) — phản ánh giả định nhân quả: **hồ sơ tài chính là nguyên nhân của quyết định duyệt/vỡ nợ, không phải hệ quả**.

**Structure Learning** — tìm $G^*$ tối ưu hóa điểm số Bayesian Information Criterion (BIC), tìm kiếm bằng Hill-Climb kết hợp Tabu List (chống local optima, chạy 50 lần độc lập theo thiết kế thực nghiệm):

$$
G^* = \arg\max_{G \in \mathcal{G}} \; \text{BIC}(G, D) = \arg\max_{G} \left[ \log L(\hat\Theta_G \mid D) - \frac{d_G}{2}\log N \right]
$$

trong đó $d_G$ là số tham số tự do của $G$, $N$ là số quan sát.

**Parameter Learning** — ước lượng hợp lý cực đại (MLE) trên từng bảng xác suất có điều kiện (CPT):

$$
\hat\theta_{ijk} = \frac{m_{ijk}}{\sum_k m_{ijk}}, \qquad \theta_{ijk} = P(X_i = k \mid \text{pa}(X_i) = j)
$$

**Exact Inference** — Variable Elimination (VE), loại bỏ tuần tự các biến ẩn bằng phép nhân-tổng (sum-product) theo thứ tự khử $\rho$:

$$
P(Y \mid E = e) = \alpha \sum_{\text{hidden}} \prod_i P(X_i \mid \text{pa}(X_i))
$$

### 2. Phân tích can thiệp (Soft-Intervention / What-if)

Với mỗi đặc trưng $X_i$, hệ thống đo lường độ nhạy của xác suất vỡ nợ khi ấn định $X_i = s$ (posterior update, tương đương *soft do*):

$$
\delta(X_i = s) = P(Y=1 \mid X_i = s) - P(Y=1)_{\text{prior}}
$$

$$
\Delta P_{\text{what-if}} = P\big(Y=1 \mid X_i = s_{\text{new}}, \text{rest}\big) - P\big(Y=1 \mid \text{profile gốc}\big)
$$

So sánh với công thức do-calculus tổng quát của Pearl:

$$
P(Y \mid do(X=x)) = \sum_{z} P(Y \mid X=x, Z=z)\, P(Z=z)
$$

— áp dụng khi $Z$ là tập biến chặn cửa sau (backdoor set). Trong phiên bản hiện tại, $P(Y \mid do(X=x)) \approx P(Y \mid X=x)$ (giả định không có confounder ẩn ngoài DAG học được), đây là **giả định đơn giản hóa** cần nêu rõ với người dùng downstream.

### 3. Diagnostic Reasoning — Likelihood Ratio

$$
LR(X_i = s) = \frac{P(X_i = s \mid Y = 1)}{P(X_i = s \mid Y = 0)}
$$

$LR > 1$: trạng thái $s$ xuất hiện nhiều hơn ở nhóm vỡ nợ → chỉ dấu rủi ro (risk discriminator).

### 4. Baseline Comparison & Explainability

Để định vị BN trong bối cảnh các phương pháp dự đoán thuần túy, dự án benchmark với **Logistic Regression, Random Forest, XGBoost, KNN, ANN (MLP), Stacking Ensemble (meta-learner = Logistic Regression, out-of-fold cross-fitting 5-fold)**, và diễn giải theo từng cá nhân bằng **LIME** (local surrogate) và **SHAP** (Shapley value) để đối chiếu với cách BN tự giải thích qua CPT.

---

## Project Architecture

```
causal-credit-risk-bn/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── configs/                 # Cấu hình forbidden_edges, siêu tham số cho từng dataset
│   ├── australian.yaml
│   ├── german_credit.yaml
│   └── lending_club.yaml
│
├── data/
│   ├── raw/                 # dữ liệu gốc 
│  
│
├── notebooks/
│   ├── australian_credit_bn.ipynb  
│   ├── german_credit_bn.ipynb       
│   └── lending_club_bn.ipynb

│
├── src/causal_credit_bn/
│   ├── preprocessing.py      # SMOTE-NC, Lasso-Stability, IV/WOE, K-Means discretization
│   ├── structure_learning.py # HillClimbSearch + Tabu + BIC + ExpertKnowledge
│   ├── parameter_learning.py # MaximumLikelihoodEstimator
│   ├── inference.py          # VariableElimination: forward / backward / what-if
│   ├── evaluation.py         # ROC-AUC, PR-AUC, Brier, ECE, Decision Curve Analysis
│   ├── explain.py            # LIME / SHAP wrapper cho bn_predict_proba
│   └── visualization.py      # Vẽ DAG (circular/target-centered), sensitivity chart
│
├── models/                   # CPT đã học (.bif/.pkl)
├── reports/figures/
├── tests/

```

---

## Installation & Quick Start

### 1. Cài đặt môi trường
```bash
git clone https://github.com/<binhba186>/causal-credit-risk-bayesian-network.git
cd causal-credit-risk-bn

python -m venv .venv
source .venv/bin/activate       

pip install -r requirements.txt
```

`requirements.txt` tối thiểu:
```
pgmpy>=0.1.25
imbalanced-learn
scikit-learn
xgboost
networkx
matplotlib
pandas
numpy
scipy
lime
shap
```

### 2. Chuẩn bị dữ liệu
```bash
mkdir -p data/raw
# Tải các bộ dữ liệu gốc vào data/raw/ (KHÔNG commit lên Git):
#   - australian.dat              (UCI Statlog Australian Credit Approval)
#   - german.data                 (UCI Statlog German Credit)
#   - lending_club_2007_2014.csv  (Kaggle Lending Club Loan Data)
```

### 3. Chạy pipeline end-to-end (ví dụ trên bộ Australian)
```bash
jupyter notebook notebooks/01_australian_credit_bn.ipynb
```

Pipeline thực thi tuần tự:

`Load & Clean → Encode → Train/Test Split → SMOTE-NC → Lasso-Stability Selection → IV/WOE Filter → K-Means Discretization (Yeo-Johnson) → BN Structure Learning (Tabu + BIC, 50 runs) → MLE Parameter Learning → Variable Elimination Inference → Evaluation (ROC/PR/DCA) → Sensitivity/Forward/Backward/What-if Analysis → Baseline Comparison (XGBoost/KNN/ANN/Stacking) → LIME/SHAP Explainability`

---

## Key Results & Practical Insights

> Bảng dưới là khung báo cáo chuẩn — **thay các ô `__` bằng số liệu thực tế đọc từ output notebook của bạn** (Step 4: Prediction Results Comparison) trước khi công bố.

### So sánh hiệu năng mô hình (Test set — Australian Credit)

| Model                     | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
|---------------------------|:--------:|:---------:|:------:|:--------:|:-------:|:------:|:-----------:|
| Bayesian Network (ours)   | __       | __        | __     | __       | __      | __     | __          |
| Logistic Regression       | __       | __        | __     | __       | __      | __     | __          |
| Random Forest             | __       | __        | __     | __       | __      | __     | __          |
| XGBoost                   | __       | __        | __     | __       | __      | __     | __          |
| KNN                       | __       | __        | __     | __       | __      | __     | __          |
| ANN (MLP)                 | __       | __        | __     | __       | __      | __     | __          |
| Stacking Ensemble         | __       | __        | __     | __       | __      | __     | __          |

*Ngưỡng phân loại: mặc định 0.5, đối chiếu thêm với ngưỡng tối ưu Youden Index/Closest-(0,1)/Symmetry lấy từ ROC trên tập train.*

### Ví dụ phân tích What-if (Counterfactual-style Intervention)

Giả sử hồ sơ vay $P_0$ có `Income = bin thấp nhất`, `CreditHistory = 0 (chưa có lịch sử tốt)`, cho kết quả baseline $P(default=1 \mid P_0) = \_\_.\_\_$.

| Kịch bản can thiệp                          | $P(default=1)$ mới | $\Delta P$ |
|----------------------------------------------|:-------------------:|:----------:|
| Baseline                                      | __                  | —          |
| `Income` → bin cao hơn 1 bậc                  | __                  | __         |
| `CreditHistory` → 1 (có lịch sử tín dụng tốt) | __                  | __         |
| Cả hai đồng thời                              | __                  | __         |

**Ý nghĩa thực tiễn**: Kết quả này gợi ý ngân hàng có thể thiết kế **chương trình cải thiện điểm tín dụng có mục tiêu** (targeted credit-improvement program) thay vì từ chối tuyệt đối — ví dụ đề xuất khách hàng xây dựng lịch sử tín dụng qua sản phẩm thẻ tín dụng đảm bảo (secured card) trước khi tái xét hồ sơ vay lớn.

### Diagnostic Insight (Backward Inference)

So sánh $P(X_i \mid Y{=}1)$ với $P(X_i \mid Y{=}0)$ và Likelihood Ratio giúp xác định **risk discriminator** hàng đầu — điền top-3 biến có $LR$ lệch xa nhất khỏi 1.0 từ output Step 7 vào đây để làm executive takeaway.

---

## Roadmap

- [ ] Hoàn thiện pipeline cho **German Credit Data** và **Lending Club 2007–2014** theo cùng kiến trúc `src/causal_credit_bn`.
- [ ] Nâng cấp từ soft-evidence sang **do-calculus đầy đủ**: xác định backdoor set tường minh, ước lượng $P(Y \mid do(X))$ bằng adjustment formula thay vì posterior update.
- [ ] Bổ sung ước lượng **ATE/CATE** bằng Double Machine Learning (DML) hoặc Inverse Propensity Weighting (IPW) làm baseline đối chiếu với BN.
- [ ] Thử nghiệm **PC Algorithm / GES** cho structure learning để so sánh độ ổn định DAG với Hill-Climb + Tabu.
- [ ] Đóng gói `bn_predict_proba` thành REST API (FastAPI) phục vụ demo credit scoring theo thời gian thực.
- [ ] Bổ sung kiểm định công bằng thuật toán (Fairness Audit: Demographic Parity, Equal Opportunity) trên các biến bị cấm làm nguyên nhân của `target`.


```

