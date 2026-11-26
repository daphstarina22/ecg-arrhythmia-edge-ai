# Preregistered Independent Evaluation Plan
## Causal Patient-Adaptive AFib Classifier with Consecutive-Window Persistence
**Document Version:** 1.0  
**Registration Date:** October 26, 2026  
**Status:** Frozen Prior to Independent Cohort Ingestion  
**Target Architecture:** Frozen Adaptive Classifier Config C + `consecutive_2` Temporal Persistence Filter  
**Safety Classification:** Research Feasibility Protocol (Not Approved for Clinical Deployment)  

---

## 1. Protocol Objective & Ethical Constraints

### 1.1 Objective
Evaluate the generalization, false-alarm suppression, and detection latency of the frozen causal patient-adaptive 5-second AFib classifier combined with a 2-window consecutive temporal persistence filter (`consecutive_2`) on a **genuinely unexposed, independent patient cohort**.

### 1.2 Strict Anti-Leakage Quarantine Protocol
1. **Quarantine of Previously Exposed Cohort**:  
   The four AFDB test patients (`04908`, `07162`, `08215`, `08219`) have already been evaluated and reported in historical studies. In strict accordance with the anti-leakage rule:
   - **These records shall NEVER be used** to report performance for this candidate.
   - **No hyperparameters, feature scalers, decision thresholds, or persistence rules** were tuned against these four records.
2. **Pre-Ingestion Freezing**:  
   This evaluation plan, along with all model weights, feature transformers, persistence state machines, and decision thresholds, is permanently frozen in this document **before** touching or extracting windows from the independent test cohort.

---

## 2. Frozen System Specification

### 2.1 Frozen Classifier Core (Configuration C)
- **Model Type**: L2-Regularized Logistic Regression (`C=1.0`, solver=`lbfgs`, max_iter=1000).
- **Training Origin**: Exclusively trained on 15 AFDB training records (`04015`, `04043`, `04048`, `04746`, `04936`, `05091`, `05261`, `06426`, `06453`, `06995`, `07162_orig`, `08378`, `08405`, `08434`, `08455`).
- **Feature Space (22 dimensions)**:
  - 11 Absolute 5-Second Features:
    1. `hr_mean`
    2. `hr_std`
    3. `rmssd`
    4. `pnn50`
    5. `sampen`
    6. `spectral_entropy`
    7. `lf_power`
    8. `hf_power`
    9. `lf_hf_ratio`
    10. `shannon_entropy`
    11. `rr_count`
  - 11 Causal Adaptive Features (Z-Score Relative to Patient Moving Baseline):
    - Moving average & standard deviation updated causally using Exponential Moving Average ($\alpha = 0.05$):
      $$\mu_t = (1 - \alpha) \mu_{t-1} + \alpha f_t$$
      $$\sigma_t^2 = (1 - \alpha) \sigma_{t-1}^2 + \alpha (f_t - \mu_t)^2$$
      $$z_t = \frac{f_t - \mu_{t-1}}{\sigma_{t-1} + \epsilon}$$
    - Baseline updates freeze immediately when alert is active to prevent baseline corruption during sustained AFib.
- **Preprocessing Scaler**: `StandardScaler` fitted strictly on the 15 training records.
- **Base Decision Threshold**: $\theta = 0.50$ (Frozen, unadjusted).

### 2.2 Frozen Temporal Persistence Filter (`consecutive_2`)
- **Filter Rule**: Alert state $y_t \in \{0, 1\}$ at window index $t$:
  $$y_t = \begin{cases} 1 & \text{if } x_t = 1 \text{ and } x_{t-1} = 1 \\ 0 & \text{otherwise} \end{cases}$$
  where $x_t = \mathbb{I}(P(\text{AFib} \mid \mathbf{z}_t) \ge 0.50)$.
- **Minimum Sustained Duration**: 2 consecutive 5-second windows ($10\text{ seconds}$).
- **Causality Constraint**: $y_t$ depends solely on current ($t$) and past ($t-1$) predictions. No lookahead is permitted.
- **Boundary Conditions**:
  - Buffer state resets to `0` upon crossing patient/record boundaries.
  - Buffer state resets to `0` if elapsed time between consecutive valid windows $\Delta t > 7.5\text{ seconds}$.
  - Rejected windows (signal quality or boundary guard bands) clear the state buffer.

---

## 3. Candidate Independent Patient Cohort

To ensure total independence from the MIT-BIH Atrial Fibrillation Database (AFDB), the preregistered evaluation cohort will be drawn from an external PhysioNet repository:

### 3.1 Primary Candidate Cohort: MIT-BIH Arrhythmia Database (MITDB)
- **Source**: PhysioNet MIT-BIH Arrhythmia Database (disjoint dataset recorded under different clinical protocols and instrumentation).
- **Sampling Rate**: $360\text{ Hz}$ (Causally resampled to $250\text{ Hz}$ to match AFDB pipeline).
- **Pre-Selected Records Containing AFib / Atrial Flutter / Arrhythmia Challenges**:
  - AFib Records: `201`, `202`, `203`, `210`, `217`, `221`, `222`.
  - Non-AFib Normal & Ectopy Controls: `100`, `101`, `103`, `105`, `106`, `119`, `200` (PVCs/Bigeminy), `207` (Ventricular Flutter/Fibrillation).
- **Ground Truth**: Extracted from official `.atr` annotation files with rhythm annotations `(AFIB`, `(AFL`, `(N`, etc.

---

## 4. Preregistered Evaluation Protocol & Metrics

### 4.1 Denominator & Rate Definitions
Both false alarm rates must be reported for each record and pooled:
1. **Rate 1 (Eligible Non-AFib Monitoring Time)**:
   $$\text{FAR}_{\text{non-AF}} = \frac{\text{False Positive Windows}}{(TN + FP) \times \frac{5}{3600}} \quad [\text{FP / hr}]$$
2. **Rate 2 (Total Eligible Clock Time)**:
   $$\text{FAR}_{\text{total}} = \frac{\text{False Positive Windows}}{\text{Total Valid Windows} \times \frac{5}{3600}} \quad [\text{FP / hr}]$$
3. **Contiguous False Alarm Event Rate**:
   $$\text{AER}_{\text{non-AF}} = \frac{\text{Contiguous Pager Alarm Events}}{\text{Eligible Non-AFib Hours}} \quad [\text{Events / hr}]$$

### 4.2 Episode-Level Evaluation
- **Episode Detection**: An episode is successfully detected if at least one persistent alert ($y_t = 1$) fires between episode start and episode end.
- **Detection Latency**:
  $$\text{Delay} = t_{\text{first\_alert}} - t_{\text{episode\_start}} \quad [\text{seconds}]$$
  Report median, mean, and 95th percentile delay.
- **Shortened/Fragmented Episodes**: Document the percentage of duration retained.

### 4.3 Success and Termination Criteria
The prospective evaluation will be considered a **SUCCESS** if and only if all the following conditions are simultaneously satisfied on the independent cohort:
1. **Pooled Episode Sensitivity**: $\ge 95.0\%$.
2. **Worst-Case Patient Episode Sensitivity**: $\ge 80.0\%$.
3. **Median Detection Latency**: $\le 10.0\text{ seconds}$.
4. **False Alarm Event Reduction**: $\ge 40.0\%$ reduction compared to unfiltered baseline.
5. **No Catastrophic Silent Misses**: Zero undetected AFib episodes exceeding $60\text{ seconds}$ in duration.

If any criterion fails, the trial terminates with a documented failure, and no parameter adjustments will be made on that cohort.

---

## 5. Execution Integrity Checklist

- [x] Adaptive baseline tracker parameters frozen ($\alpha = 0.05, \epsilon = 10^{-4}$).
- [x] Feature scaler parameters frozen.
- [x] Classifier weights and threshold frozen ($\theta = 0.50$).
- [x] Persistence filter rule frozen (`consecutive_2`, $k=2$).
- [x] Historical test records quarantined (`04908`, `07162`, `08215`, `08219`).
- [x] Independent cohort identified (MITDB).
- [x] Success criteria preregistered.

*Preregistration confirmed by Antigravity AI Engineering Agent.*
