# MedGemma 27B Model Test Report

**Generated:** 2026-01-15
**Test Date:** 2026-01-15T15:13:11
**Total Cases:** 10

---

## 1. Environment Configuration

### Infrastructure
| Component | Specification |
|-----------|---------------|
| Cloud Provider | Google Cloud Platform (GCP) |
| VM Instance | medgemma-llamacpp |
| Zone | us-west1-a |
| Machine Type | g2-standard-12 |
| GPU | NVIDIA L4 (24GB VRAM) |

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Model | MedGemma 27B Instruct |
| Source | unsloth/medgemma-27b-it-GGUF |
| Quantization | Q4_K_M (~16GB) |
| Inference Server | Ollama |
| Context Length | 4096 tokens |
| Keep Alive | 4 hours |

### Access Method
- **Network:** No external IP (secure deployment)
- **Access:** Google IAP (Identity-Aware Proxy) tunnel
- **Local Endpoint:** http://localhost:18080

---

## 2. Test Summary

### Overview
All 10 patient cases were processed successfully. MedGemma provided comprehensive differential diagnoses with detailed clinical reasoning for each case.

### Response Statistics
| Metric | Value |
|--------|-------|
| Total Cases | 10 |
| Successful | 10 |
| Failed | 0 |
| Avg Response Length | ~7,500 characters |
| Min Response Length | 5,922 characters (Case 6) |
| Max Response Length | 11,373 characters (Case 3) |

---

## 3. Test Cases Analysis

### Case 1: Acute Coronary Syndrome
| Field | Value |
|-------|-------|
| Patient | 45-year-old male |
| Symptoms | Chest pain radiating to left arm, shortness of breath, sweating |
| Response Length | 7,156 characters |

**Primary Diagnosis:** Acute Myocardial Infarction (AMI)

**Key Differentials Identified:**
1. Acute Myocardial Infarction (AMI) - *High Priority*
2. Unstable Angina - *High Priority*
3. Pulmonary Embolism - *High Priority*
4. Aortic Dissection - *High Priority*
5. Pericarditis, Pneumothorax, Esophageal causes

**Assessment:** ✅ Excellent response. Correctly identified cardiac emergency with appropriate urgency. Comprehensive differential including all life-threatening causes.

---

### Case 2: Congestive Heart Failure
| Field | Value |
|-------|-------|
| Patient | 62-year-old male |
| Symptoms | Leg swelling, shortness of breath when lying down, fatigue |
| Response Length | 8,432 characters |

**Primary Diagnosis:** Congestive Heart Failure (CHF)

**Key Differentials Identified:**
1. Congestive Heart Failure (CHF) - *Most Likely*
2. Chronic Kidney Disease (CKD)
3. Liver Disease (Cirrhosis)
4. Venous Insufficiency
5. Medication Side Effects, Hypothyroidism, Nephrotic Syndrome

**Assessment:** ✅ Excellent response. Correctly identified orthopnea as key symptom. Comprehensive consideration of fluid overload causes.

---

### Case 3: Cardiac Arrhythmia/Syncope
| Field | Value |
|-------|-------|
| Patient | 58-year-old female |
| Symptoms | Palpitations, dizziness, occasional fainting spells |
| Response Length | 11,373 characters |

**Primary Diagnosis:** Cardiac Arrhythmias (SVT, AFib, VT, Bradyarrhythmias)

**Key Differentials Identified:**
1. Arrhythmias (multiple types) - *Cardiovascular*
2. Structural Heart Disease
3. Vasovagal Syncope - *Neurologic*
4. Orthostatic Hypotension
5. Metabolic causes (Hypoglycemia, Anemia, Thyroid)

**Assessment:** ✅ Excellent response. Most comprehensive analysis with detailed summary table. Properly categorized by system (cardiovascular, neurologic, metabolic, psychiatric).

---

### Case 4: Stable Angina
| Field | Value |
|-------|-------|
| Patient | 70-year-old male |
| Symptoms | Chest tightness during exertion, relieved by rest |
| Response Length | 6,424 characters |

**Primary Diagnosis:** Stable Angina Pectoris

**Key Differentials Identified:**
1. Stable Angina Pectoris - *Most Likely*
2. Unstable Angina - *Must Rule Out*
3. Myocardial Infarction - *Must Rule Out*
4. Non-cardiac causes (musculoskeletal, GI, pulmonary)

**Assessment:** ✅ Excellent response. Classic exertional angina correctly identified. Appropriately emphasized need to rule out acute coronary syndrome.

---

### Case 5: Hypertensive Emergency
| Field | Value |
|-------|-------|
| Patient | 55-year-old female |
| Symptoms | Severe headache, nosebleeds, blurred vision, dizziness |
| Response Length | 6,656 characters |

**Primary Diagnosis:** Hypertensive Emergency/Urgency

**Key Differentials Identified:**
1. Hypertensive Emergency - *High Priority*
2. Intracranial Hemorrhage - *High Priority*
3. Stroke - *High Priority*
4. Giant Cell Arteritis
5. Brain Tumor, Sinusitis, Migraine

**Assessment:** ✅ Excellent response. Correctly prioritized blood pressure check as immediate action. Comprehensive consideration of serious neurological causes.

---

### Case 6: Meningitis
| Field | Value |
|-------|-------|
| Patient | 32-year-old female |
| Symptoms | Severe headache, sensitivity to light, nausea, stiff neck |
| Response Length | 5,922 characters |

**Primary Diagnosis:** Meningitis

**Key Differentials Identified:**
1. Meningitis - *Most Likely & Serious*
2. Subarachnoid Hemorrhage - *Most Likely & Serious*
3. Migraine (less likely)
4. Encephalitis
5. Cerebral Venous Sinus Thrombosis

**Assessment:** ✅ Excellent response. Correctly identified classic meningitis triad. Appropriately emphasized bacterial meningitis as primary concern requiring urgent treatment.

---

### Case 7: Dementia
| Field | Value |
|-------|-------|
| Patient | 72-year-old male |
| Symptoms | Memory problems, confusion, difficulty finding words, getting lost |
| Response Length | 6,975 characters |

**Primary Diagnosis:** Alzheimer's Disease

**Key Differentials Identified:**
1. Alzheimer's Disease - *Most Common*
2. Vascular Dementia
3. Lewy Body Dementia
4. Frontotemporal Dementia
5. Reversible causes (delirium, depression, medications, metabolic)

**Assessment:** ✅ Excellent response. Comprehensive dementia workup outlined. Appropriately emphasized ruling out reversible causes before diagnosing neurodegenerative disease.

---

### Case 8: Acute Stroke
| Field | Value |
|-------|-------|
| Patient | 75-year-old male |
| Symptoms | Sudden weakness on right side, slurred speech, facial droop |
| Response Length | 7,431 characters |

**Primary Diagnosis:** Acute Ischemic Stroke

**Key Differentials Identified:**
1. Acute Ischemic Stroke - *Most Likely*
2. Transient Ischemic Attack (TIA)
3. Hemorrhagic Stroke
4. Seizure (Todd's Paralysis)
5. Brain Tumor, Migraine, Hypoglycemia

**Assessment:** ✅ Excellent response. Correctly identified stroke as medical emergency. Comprehensive immediate action steps including stroke protocol activation.

---

### Case 9: Parkinson's Disease
| Field | Value |
|-------|-------|
| Patient | 45-year-old female |
| Symptoms | Tremor at rest, slow movement, rigid muscles, stooped posture |
| Response Length | 7,135 characters |

**Primary Diagnosis:** Parkinson's Disease

**Key Differentials Identified:**
1. Parkinson's Disease - *Most Likely*
2. Atypical Parkinsonism (MSA, PSP, CBD, DLB)
3. Drug-Induced Parkinsonism
4. Secondary Parkinsonism (vascular, post-encephalitic)
5. Essential Tremor (less likely)

**Assessment:** ✅ Excellent response. Classic parkinsonian tetrad correctly identified. Appropriately noted early-onset PD consideration given patient's age (45).

---

### Case 10: Multiple Sclerosis
| Field | Value |
|-------|-------|
| Patient | 28-year-old female |
| Symptoms | Episodes of vision loss, numbness in limbs, fatigue, balance problems |
| Response Length | 8,305 characters |

**Primary Diagnosis:** Multiple Sclerosis (MS)

**Key Differentials Identified:**
1. Multiple Sclerosis - *Highly Likely*
2. Neuromyelitis Optica Spectrum Disorder (NMOSD)
3. Acute Disseminated Encephalomyelitis (ADEM)
4. Conversion Disorder (functional)
5. Vasculitis, Vitamin B12 Deficiency, SLE

**Assessment:** ✅ Excellent response. Classic MS presentation in typical demographic (young female) correctly identified. Comprehensive discussion of demyelinating diseases.

---

## 4. Quality Assessment

### Clinical Reasoning Quality

| Criteria | Score | Notes |
|----------|-------|-------|
| Diagnostic Accuracy | 10/10 | All primary diagnoses were clinically appropriate |
| Differential Breadth | 10/10 | Comprehensive differentials for each case |
| Clinical Reasoning | 10/10 | Clear explanation of symptom-diagnosis correlation |
| Urgency Recognition | 10/10 | Correctly identified emergencies (MI, stroke, meningitis) |
| Next Steps | 10/10 | Appropriate diagnostic workup recommendations |

### Response Characteristics

- **Structured Format:** All responses used consistent formatting with headers, bullet points, and tables
- **Medical Terminology:** Appropriate use of clinical terminology with explanations
- **Disclaimers:** All responses included appropriate medical disclaimers
- **Completeness:** Responses included history, examination, and investigation recommendations

---

## 5. Conclusions

### Strengths
1. **Accurate Primary Diagnoses:** MedGemma correctly identified the most likely diagnosis in all 10 cases
2. **Comprehensive Differentials:** Each response included multiple differential diagnoses organized by likelihood and urgency
3. **Clinical Reasoning:** Detailed explanations connecting symptoms to diagnoses
4. **Emergency Recognition:** Appropriately identified and prioritized life-threatening conditions
5. **Actionable Recommendations:** Provided clear next steps for evaluation and management

### Observations
1. Response times varied based on complexity (timeout occurred on 1 retry for Case 1)
2. Longer responses correlated with cases having broader differentials (e.g., Case 3: syncope)
3. All responses maintained consistent structure and quality

### Recommendations
1. Model performs well for general medical differential diagnosis
2. Suitable for clinical decision support applications
3. Consider for medical education and training scenarios
4. Always include appropriate disclaimers for clinical use

---

## 6. Appendix

### Test Configuration
```python
# API Configuration
OLLAMA_URL = "http://localhost:18080/api/chat"
MODEL = "medgemma"
TIMEOUT = 300 seconds

# System Prompt
"You are a helpful medical assistant. Provide differential diagnoses based on patient symptoms."
```

### Sample Request Format
```json
{
  "model": "medgemma",
  "messages": [
    {"role": "system", "content": "You are a helpful medical assistant..."},
    {"role": "user", "content": "Patient Information:\n- Age: X\n- Gender: Y\n- Symptoms: Z\n\nBased on these symptoms, what are the possible differential diagnoses?"}
  ],
  "stream": false
}
```

---

*Report generated by MedGemma Demo Test Suite*
