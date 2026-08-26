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

### Cấu hình Thuật toán Tối ưu Cấu trúc (Tabu Search)

* **Số lần huấn luyện (`N_RUNS`):** `50`
* **Bộ nhớ Tabu (`tabu_length`):** `100` – Hoạt động như một bộ nhớ ngắn hạn lưu trữ các hành động thêm, xóa, đảo ngược cạnh mà thuật toán vừa thực hiện trong 100 bước gần nhất.
* **Bậc vào tối đa mỗi node (`max_indegree`):** `5` – Số mũi tên tối đa hướng trực tiếp đi vào một nút (tương ứng số nút cha tối đa), giúp kiểm soát kích thước bảng CPT.
* **Số bước tìm kiếm tối đa mỗi run (`max_iter`):** `1000`.
* **Xác suất sinh cạnh ban đầu (`edge_prob`):** `0.3` – Đặt xác suất thấp ($\leq 0.5$) để tránh đồ thị sinh ra quá dày đặc, làm tăng độ phức tạp. Đối với đồ thị có hướng không chu trình (DAG) gồm $n$ nút, số lượng cặp nút tối đa có thể thiết lập liên kết là $M_{\text{max}} = \binom{n}{2} = \frac{n(n-1)}{2}$. Số lượng cạnh trung bình xuất hiện trên đồ thị ban đầu được tính bằng:

$$
E(M) = \text{edge\\_prob} \times M_{\text{max}} = \text{edge\\_prob} \times \frac{n(n-1)}{2}
$$

---

### Kết quả Học Cấu trúc DAG

Sau 50 lần chạy, mô hình DAG có điểm BIC cao nhất (ít âm nhất) được chọn làm cấu trúc cuối cùng.

**Bảng kết quả tối ưu cấu trúc DAG bằng thuật toán Tabu Search**

| Bộ dữ liệu | BIC tốt nhất | Run tốt nhất | Số node | Số cạnh | Dao động BIC giữa các run |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Australian Credit** | -3,343.32 | 15 | 9 | 10 | $\approx 44$ đơn vị (-3387 $\rightarrow$ -3343) |
| **German Credit** | -14,755.32 | 20 | 16 | 18 | $\approx 114$ đơn vị (-14869 $\rightarrow$ -14755) |
| **Lending Club** | -725,970.15 | 14 | 28 | 58 | $\approx 3,500$ đơn vị (-729492 $\rightarrow$ -725970) |

---
<p align="center">
    <img width="850" height="652" alt="image" src="https://github.com/user-attachments/assets/ea4ed19b-b83e-4788-a0d3-545aaecaf7b0" />
<br>
  <em></b> Sơ đồ cấu trúc mạng Bayes Australian Credit dataset</em>
</p>

<p align="center">
<img width="1042" height="808" alt="image" src="https://github.com/user-attachments/assets/a35a7ffb-b188-4802-a9c0-eca2a7bbe5b8" />
<br>
  <em></b> Sơ đồ cấu trúc mạng Bayes German Credit dataset</em>
</p>

<p align="center">
<img width="1081" height="847" alt="image" src="https://github.com/user-attachments/assets/597da165-fee9-4d92-9c9e-2d03cbdece17" />
<br>
  <em></b> Sơ đồ cấu trúc mạng Bayes Lending Club dataset</em>
</p>

### Đánh giá Mô hình trên Tập Huấn luyện (Train) và Kiểm tra (Test)

**Bảng tổng hợp chỉ số đánh giá mô hình Mạng Bayes (Bayesian Network)**

| Chỉ số | Lending Club (Train) | Lending Club (Test) | German Credit (Train) | German Credit (Test) | Australian Credit (Train) | Australian Credit (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.7085 | 0.6808 | 0.7811 | 0.7300 | 0.8641 | 0.8406 |
| **Precision** | 0.7269 | 0.4610 | 0.8386 | 0.6047 | 0.9247 | 0.9211 |
| **Recall** | 0.6679 | 0.4878 | 0.6962 | 0.4127 | 0.7926 | 0.8140 |
| **Specificity** | 0.7490 | 0.7615 | 0.8660 | 0.8759 | 0.9355 | 0.8846 |
| **F1-Score** | 0.7034 | 0.4740 | 0.7608 | 0.4906 | 0.8536 | 0.8642 |
| **ROC-AUC** | 0.7944 | 0.6910 | 0.8686 | 0.7667 | 0.9278 | 0.9220 |
| **PR-AUC** | 0.8238 | 0.5224 | 0.8886 | 0.5421 | 0.9070 | 0.9371 |

<p align="center">
    <img width="850" height="652" alt="image" src="https://github.com/user-attachments/assets/b25c1d78-e3d1-48d2-b347-bdc7fe9fa9ef" />
<br>
  <em></b> Phân phối xác suất của toàn mạng --- Australian Credit</em>
</p>


<p align="center">
<img width="1042" height="808" alt="image" src="https://github.com/user-attachments/assets/96c51994-5239-4644-a50c-fb2a898a5c1f" />
<br>
  <em></b> Phân phối xác suất của toàn mạng --- German Credit</em>
</p>

<p align="center">
<img width="1081" height="847" alt="image" src="https://github.com/user-attachments/assets/7a3ab662-ff7f-478a-ae8f-1845aec36135" />
<br>
  <em></b> Phân phối xác suất của toàn mạng --- Lending Club</em>
</p>


#### Phân tích Đường cong ROC, Ngưỡng Tối ưu và Decision Curve Analysis (DCA) — Australian Credit

<p align="center">
  <img src="https://github.com/user-attachments/assets/d9ed8276-dc27-47ad-95e5-637ff3252a8d" alt="ROC Curve Australian Credit" width="75%">
  <br>
  <em>Đồ thị ba ngưỡng quyết định tối ưu theo ROC — Australian Credit</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/08401c18-c7ce-4780-a9eb-4d574e90a8be" alt="DCA Curve Australian Credit" width="75%">
  <br>
  <em>Đồ thị ba ngưỡng quyết định tối ưu trên DCA — Australian Credit</em>
</p>


---

**Bảng ba ngưỡng quyết định tối ưu theo ROC — Australian Credit (Tập Validation)**

| Tiêu chí ngưỡng | Ngưỡng ($c$) | Sensitivity | Specificity | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Youden Index** | 0.4254 | 0.8125 | 0.9138 | 0.8551 | 0.8667 |
| **Closest to (0,1)** | 0.4052 | 0.8125 | 0.9138 | 0.8551 | 0.8667 |
| **Symmetry Point** | 0.3818 | 0.9250 | 0.7241 | 0.8406 | 0.8706 |

> **Đánh giá DCA:** Với tỷ lệ lưu hành (prevalence) = 0.5797, ngưỡng **Symmetry Point** ($c_S = 0.3818$) mang lại Lợi ích ròng (Net Benefit) cao nhất trong ba ngưỡng với $\text{NB} = 0.4661$, vượt trội so với chiến lược chấp thuận toàn bộ (Treat All) **0.1459 điểm**.

**So sánh chỉ số đánh giá mô hình Mạng Bayes với ngưỡng tối ưu trên tập Test**

| Chỉ số | Australian ($c=0.5$) | Australian ($c^*=0.3818$) | German ($c=0.5$) | German ($c^*=0.1895$) | Lending Club ($c=0.5$) | Lending Club ($c^*=0.3593$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.8406 | **0.8551** | **0.7300** | 0.6750 | **0.6808** | 0.5897 |
| **Precision** | **0.9211** | 0.8367 | **0.6047** | 0.4891 | **0.4610** | 0.3926 |
| **Recall** | 0.8140 | **0.9535** | 0.4127 | **0.7143** | 0.4878 | **0.7157** |
| **Specificity** | **0.8846** | 0.6923 | **0.8759** | 0.6569 | **0.7615** | 0.5370 |
| **F1-Score** | 0.8642 | **0.8913** | 0.4906 | **0.5806** | 0.4740 | **0.5070** |

### So sánh mạng Bayes với các mô hình máy học đối chứng
<p align="center">
  <img src="https://github.com/user-attachments/assets/ecd3a944-bf26-4337-9c6e-253dbbac2799" alt="all model German Credit" width="75%">
  <br>
  <em>So sánh hiệu năng của mô hình mạng Bayes với các mô hình máy học trên bộ dữ liệu German Credit</em>
</p>
<img width="2316" height="1751" alt="auc_ger" src="https://github.com/user-attachments/assets/ecd3a944-bf26-4337-9c6e-253dbbac2799" />

### Sensitivity analysis
<img width="632" height="862" alt="image" src="https://github.com/user-attachments/assets/e6e5755e-cbc2-4cba-bbeb-683d1b614f62" />

### Forward Inference
<img width="1055" height="654" alt="image" src="https://github.com/user-attachments/assets/39ecaa45-7c63-4153-b493-1efec1f54622" />

### Diagnostic Insight (Backward Inference)

<img width="687" height="731" alt="image" src="https://github.com/user-attachments/assets/d75c3dbd-f967-4259-a582-a3aea22b7d45" />

<p align="center">
    <img width="687" height="753" alt="image" src="https://github.com/user-attachments/assets/64a07fe0-a7d1-4db9-84cc-ff758201d1c8" />
<br>
  <em></b>  Suy luận lùi toàn cục theo evidence target — German Credit </em>
</p>


### Phân tích kịch bản giả định (what-if)
<p align="center">
    <img width="1037" height="742" alt="image" src="https://github.com/user-attachments/assets/d51238d2-e499-4655-8da6-eab98588eb6a" />
<br>
  <em></b>  Phân tích kịch bản giả định theo hồ sơ cá nhân --- Australian Credit {\newline \small (Trong đó, màu xanh: target = 0 và màu đỏ: target = 1})</em>
</p>

<p align="center"><b>CPT của target theo Employed × Income — Australian Credit</b></p>

<table align="center">
  <thead>
    <tr>
      <th><b>Employed</b></th>
      <th>0</th><th>0</th><th>0</th><th>0</th>
      <th>1</th><th>1</th><th>1</th><th>1</th>
    </tr>
    <tr>
      <th><b>Income</b></th>
      <th>0</th><th>1</th><th>2</th><th>3</th>
      <th>0</th><th>1</th><th>2</th><th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>P(target=1)</b></td>
      <td>0.7190</td><td>0.7474</td><td>0.7871</td><td>0.6174</td>
      <td>0.2455</td><td>0.6606</td><td>0.3026</td><td>0.0441</td>
    </tr>
    <tr>
      <td><b>P(target=0)</b></td>
      <td>0.2810</td><td>0.2526</td><td>0.2129</td><td>0.3826</td>
      <td>0.7545</td><td>0.3394</td><td>0.6974</td><td>0.9559</td>
    </tr>
  </tbody>
</table>

### Đối chiếu với công cụ diễn giải hậu kỳ riêng biệt (như SHAP, LIME)

<p align="center">
    <img width="770" height="862" alt="image" src="https://github.com/user-attachments/assets/72fd8b50-afe2-4e29-8595-c272aa00d78c" />
<br>
  <em></b>   So sánh ba phương pháp giải thích cục bộ cho một hồ sơ khách hàng ngẫu nhiên — German Credit</em>
</p>
---

## Roadmap

- [ ] Hoàn thiện pipeline cho **German Credit Data** và **Lending Club 2007–2014** theo cùng kiến trúc `src/causal_credit_bn`.
- [ ] Nâng cấp từ soft-evidence sang **do-calculus đầy đủ**: xác định backdoor set tường minh, ước lượng $P(Y \mid do(X))$ bằng adjustment formula thay vì posterior update.
- [ ] Bổ sung ước lượng **ATE/CATE** bằng Double Machine Learning (DML) hoặc Inverse Propensity Weighting (IPW) làm baseline đối chiếu với BN.
- [ ] Thử nghiệm **PC Algorithm / GES** cho structure learning để so sánh độ ổn định DAG với Hill-Climb + Tabu.
- [ ] Đóng gói `bn_predict_proba` thành REST API (FastAPI) phục vụ demo credit scoring theo thời gian thực.
- [ ] Bổ sung kiểm định công bằng thuật toán (Fairness Audit: Demographic Parity, Equal Opportunity) trên các biến bị cấm làm nguyên nhân của `target`.


```

